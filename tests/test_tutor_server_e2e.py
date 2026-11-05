"""End-to-end test of the real HTTP server, without a model or an API key.

The in-process server tests in ``test_tutor_server.py`` exercise the handler, but
they do not exercise the command a school actually runs, the environment it
inherits, or the process shutdown. This test starts ``python -m
lligma.tutor.server --backend extractive`` as a child process with
``OPENAI_API_KEY`` removed, waits for ``/api/health`` instead of sleeping, asks
the two historically difficult evaluation questions, and checks that two
out-of-scope questions still abstain. A second test drives ``/api/stream`` and
reconstructs the answer from its SSE events, so the streaming endpoint is held
to the same grounding and abstention guarantees as ``/api/ask``. A third test
closes the socket with an RST after the first SSE event and checks that the
server still serves health, stream and ask requests without an unhandled
traceback. A test-only delay keeps an event pending, and the additive
``stream_aborts`` health counter proves that the failed-write path ran. A fourth
test keeps one hundred RST clients open together with a complete SSE client and
checks the exact ``stream_aborts`` delta plus the byte-for-byte cached response.
The child is started through ``tests/tutor_server_instrumented.py``, a test-only
entrypoint that monkeypatches only that process: a barrier aligns the hundred
abort handlers, an opcode trace sleeps just before the ``STORE_ATTR`` in
``_note_stream_abort``, and a wrapper delays SSE writes. The production server
keeps only the real lock around the counter.
The trace does not alter events, so the healthy payload must stay byte-identical.
The repeatable experiment is ``python tests/mutation_stream_aborts.py``. In
three consecutive runs it reported 15/15 protected stream runs with exact delta
100; the mutated stream runs failed 9/9, with observed ``stream_aborts`` values
3, 4, 4; 2, 3, 2; and 3, 3, 4. A later WMI-watchdog verification observed
96, 86 and 89, still below the required 100. The same command mutates the
second protected invariant, ``AnswerCache._lock``: 15/15 pytest runs of
``tests/test_tutor_service.py`` reported 23/23, while the lock-removal mutation
failed 9/9 with 1 failed and 22 passed, always on
``test_cache_serializes_concurrent_mutations``. The command then runs two real
pytest subprocesses on ``tests/mutation_fault_case.py``: one raises an explicit
``RuntimeError`` (pytest exit 1), the other raises ``KeyboardInterrupt`` during
the run (pytest exit 2). The ``finally`` in ``tests/mutation_fault_harness.py``
and ``tests/mutation_cache_lock.py`` restores the production files
byte-identically, with locks present and mutations absent.

The early-disconnect test intentionally sends RST. ``BaseHTTPRequestHandler``
may then try to read the next request and ``socketserver`` logs a
``ConnectionResetError`` traceback. The test-only entrypoint consumes only that
expected reset in ``handle_one_request``; all other exceptions still propagate,
so the existing payload, abort and log assertions remain meaningful. After this
fix, 25/25 complete E2E runs passed before the three consecutive full-command
runs above.

The port is chosen by the OS and the child is terminated through its process
group (``taskkill /T`` on Windows, ``killpg`` elsewhere). Cleanup runs from a
``finally``-equivalent context manager and verifies that the port is no longer
listening, so a failed assertion cannot leave a server behind.

Run with pytest, or directly: python tests/test_tutor_server_e2e.py
"""

from __future__ import annotations

import http.client
import json
import os
import re
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lligma.tutor.text import normalize  # noqa: E402

READY_TIMEOUT_SECONDS = 10.0
STOP_TIMEOUT_SECONDS = 5.0
POLL_INTERVAL_SECONDS = 0.05
STREAM_TEST_DELAY_ENV = "LLIGMA_TUTOR_TEST_SSE_DELAY_MS"
STREAM_TEST_DELAY_MS = "100"
CONCURRENT_ABORTS = 100
CONCURRENT_STREAM_DELAY_MS = "1000"
CONCURRENT_SWITCH_INTERVAL_ENV = "LLIGMA_TUTOR_TEST_SWITCH_INTERVAL"
CONCURRENT_SWITCH_INTERVAL = "0.0000001"
CONCURRENT_ABORT_YIELD_ENV = "LLIGMA_TUTOR_TEST_ABORT_YIELD_MS"
CONCURRENT_ABORT_YIELD_MS = "1"
CONCURRENT_ABORT_BARRIER_ENV = "LLIGMA_TUTOR_TEST_ABORT_BARRIER"
CONCURRENT_ABORT_BARRIER = str(CONCURRENT_ABORTS)


def free_port() -> int:
    """Ask the OS for a free loopback port instead of assuming 8080."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def port_is_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def load_eval_rows() -> list[dict]:
    with (ROOT / "data" / "tutor" / "eval.jsonl").open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_sse_events(raw: bytes) -> list[dict]:
    """Estrae gli eventi da un corpo SSE gia' letto per intero."""
    events = []
    for block in raw.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        payload = "\n".join(
            line[5:].lstrip() for line in block.split("\n") if line.startswith("data:")
        )
        if payload:
            events.append(json.loads(payload))
    return events


class RunningServer:
    """A real server child, optionally started through the test entrypoint.

    The production module is used by default. ``instrumented=True`` starts
    ``tests/tutor_server_instrumented.py``, which monkeypatches the counter and
    the SSE writer only inside that child process.
    """

    def __init__(
        self,
        env_overrides: dict[str, str] | None = None,
        instrumented: bool = False,
    ):
        self.port = free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.env_overrides = dict(env_overrides or {})
        self.instrumented = instrumented
        self.process: subprocess.Popen | None = None
        self.log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        self._stopped = False

    def __enter__(self):
        try:
            self.start()
        except BaseException:
            self.stop()
            raise
        return self

    def __exit__(self, *exc_info):
        self.stop()

    def start(self) -> None:
        env = os.environ.copy()
        # The whole point of this test is the path with no model/API key.
        env.pop("OPENAI_API_KEY", None)
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env.update(self.env_overrides)
        assert "OPENAI_API_KEY" not in env, "la chiave API non e' stata rimossa dall'ambiente"

        command = [sys.executable]
        if self.instrumented:
            command.append(str(ROOT / "tests" / "tutor_server_instrumented.py"))
        else:
            command.extend(["-m", "lligma.tutor.server"])
        command.extend(
            [
                "--host",
                "127.0.0.1",
                "--port",
                str(self.port),
                "--backend",
                "extractive",
            ]
        )
        kwargs = {
            "cwd": str(ROOT),
            "env": env,
            "stdout": self.log,
            "stderr": subprocess.STDOUT,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
        if os.name == "nt":
            # A separate group is useful to taskkill /T; taskkill remains the
            # reliable way to include any child the server may have created.
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        try:
            self.process = subprocess.Popen(command, **kwargs)
        except OSError as error:
            raise AssertionError(f"impossibile avviare il server: {error}") from error
        self._wait_until_ready()

    def _output(self) -> str:
        self.log.flush()
        self.log.seek(0)
        return self.log.read().strip()

    def _wait_until_ready(self) -> dict:
        assert self.process is not None
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            returncode = self.process.poll()
            if returncode is not None:
                raise AssertionError(
                    f"il server e' terminato con codice {returncode} prima di essere pronto.\n"
                    f"output:\n{self._output()}"
                )
            try:
                with urllib.request.urlopen(self.base + "/api/health", timeout=0.5) as response:
                    if response.status != 200:
                        last_error = RuntimeError(f"health HTTP {response.status}")
                        continue
                    health = json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as error:
                last_error = error
            else:
                assert health.get("backend") == "extractive", (
                    f"il server non e' in modalita' extractive: {health!r}"
                )
                assert health.get("status") == "ok", f"health inatteso: {health!r}"
                return health
            time.sleep(POLL_INTERVAL_SECONDS)

        raise AssertionError(
            f"server non pronto entro {READY_TIMEOUT_SECONDS:.0f}s; "
            f"ultimo errore: {last_error!r}.\noutput:\n{self._output()}"
        )

    def health(self) -> dict:
        try:
            with urllib.request.urlopen(self.base + "/api/health", timeout=5) as response:
                assert response.status == 200, f"health HTTP {response.status}"
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise AssertionError(f"health fallito: {error}") from error

    def output(self) -> str:
        return self._output()

    def wait_for_stream_aborts(self, expected: int = 1, timeout: float = STOP_TIMEOUT_SECONDS) -> dict:
        """Attende il segnale diagnostico di almeno ``expected`` abort SSE."""
        deadline = time.monotonic() + timeout
        last_error: BaseException | None = None
        while time.monotonic() < deadline:
            try:
                health = self.health()
            except AssertionError as error:
                last_error = error
            else:
                aborts = int(health.get("stream_aborts") or 0)
                if aborts >= expected:
                    return health
                last_error = AssertionError(f"stream_aborts={aborts}, atteso almeno {expected}")
            time.sleep(POLL_INTERVAL_SECONDS)
        raise AssertionError(
            f"meno di {expected} abort SSE registrati entro {timeout:.1f}s; "
            f"ultimo stato: {last_error!r}.\noutput:\n{self._output()}"
        )

    def ask(self, question: str) -> dict:
        request = urllib.request.Request(
            self.base + "/api/ask",
            data=json.dumps({"question": question}, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                assert response.status == 200, f"HTTP {response.status} per {question!r}"
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", "replace")
            raise AssertionError(f"HTTP {error.code} per {question!r}: {body}") from error
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as error:
            raise AssertionError(f"richiesta fallita per {question!r}: {error}") from error

    def stream(self, question: str) -> tuple[int, str, str, list[dict]]:
        """Legge ``/api/stream`` riga per riga, come farebbe un client SSE."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
            connection.request(
                "POST",
                "/api/stream",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            status = response.status
            content_type = response.getheader("Content-Type", "")
            cache_control = response.getheader("Cache-Control", "")
            if status != 200:
                detail = response.read().decode("utf-8", "replace")
                raise AssertionError(f"HTTP {status} per lo stream di {question!r}: {detail}")

            raw_lines: list[str] = []
            while True:
                line = response.readline()
                if not line:
                    break
                raw_lines.append(line.decode("utf-8", "replace"))
        except (OSError, http.client.HTTPException) as error:
            raise AssertionError(f"stream fallito per {question!r}: {error}") from error
        finally:
            connection.close()

        events: list[dict] = []
        pending: list[str] = []
        for raw_line in raw_lines:
            line = raw_line.rstrip("\r\n")
            if not line:
                if pending:
                    payload = "\n".join(pending)
                    try:
                        events.append(json.loads(payload))
                    except json.JSONDecodeError as error:
                        raise AssertionError(f"evento SSE non valido per {question!r}: {payload!r}") from error
                    pending = []
                continue
            if line.startswith(":"):
                continue
            if not line.startswith("data:"):
                raise AssertionError(f"riga SSE inattesa per {question!r}: {line!r}")
            pending.append(line[5:].lstrip())
        if pending:
            raise AssertionError(f"evento SSE non terminato da una riga vuota per {question!r}")
        assert events, f"nessun evento SSE per {question!r}"
        return status, content_type, cache_control, events

    def stream_raw(self, question: str) -> tuple[int, str, str, bytes]:
        """Legge una risposta SSE completa e restituisce il corpo esatto."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
            connection.request(
                "POST",
                "/api/stream",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            status = response.status
            content_type = response.getheader("Content-Type", "")
            cache_control = response.getheader("Cache-Control", "")
            raw = response.read()
            if status != 200:
                raise AssertionError(
                    f"HTTP {status} per lo stream di {question!r}: "
                    f"{raw.decode('utf-8', 'replace')}"
                )
            return status, content_type, cache_control, raw
        except (OSError, http.client.HTTPException) as error:
            raise AssertionError(f"stream fallito per {question!r}: {error}") from error
        finally:
            connection.close()

    def open_stream_socket(
        self, question: str
    ) -> tuple[socket.socket, int, str, bytes, dict]:
        """Apre uno stream, legge header e primo evento, e lascia la socket aperta."""
        body = json.dumps({"question": question}, ensure_ascii=False).encode("utf-8")
        request = (
            "POST /api/stream HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{self.port}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii") + body
        sock = socket.create_connection(("127.0.0.1", self.port), timeout=10)
        try:
            sock.sendall(request)
            header_bytes = b""
            while not header_bytes.endswith(b"\r\n\r\n"):
                chunk = sock.recv(1)
                if not chunk:
                    raise AssertionError("il server ha chiuso prima degli header SSE")
                header_bytes += chunk
            header_lines = header_bytes[:-4].decode("latin-1").split("\r\n")
            status = int(header_lines[0].split()[1])
            headers = {}
            for line in header_lines[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    headers[key.strip().lower()] = value.strip()
            content_type = headers.get("content-type", "")

            first_event_bytes = b""
            while not first_event_bytes.endswith(b"\n\n"):
                chunk = sock.recv(1)
                if not chunk:
                    raise AssertionError("il server ha chiuso prima del primo evento SSE")
                first_event_bytes += chunk
            first_event = first_event_bytes[:-2]
            payload_lines = [
                line[5:].lstrip()
                for line in first_event.decode("utf-8").replace("\r\n", "\n").split("\n")
                if line.startswith("data:")
            ]
            assert payload_lines, f"primo evento SSE senza data: {first_event!r}"
            event = json.loads("\n".join(payload_lines))
        except BaseException:
            sock.close()
            raise
        return sock, status, content_type, first_event_bytes, event

    @staticmethod
    def close_with_rst(sock: socket.socket) -> None:
        """Chiude con RST: SO_LINGER=0 evita un semplice FIN."""
        linger = struct.pack("hh", 1, 0) if os.name == "nt" else struct.pack("ii", 1, 0)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, linger)
        except OSError:
            pass
        sock.close()

    def read_stream_rest(self, sock: socket.socket, first_event_bytes: bytes) -> bytes:
        """Legge il resto di uno stream aperto, incluso il terminatore done."""
        sock.settimeout(10)
        chunks = [first_event_bytes]
        try:
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                chunks.append(chunk)
        except OSError as error:
            raise AssertionError(f"lettura dello stream interrotta: {error}") from error
        return b"".join(chunks)

    def disconnect_mid_stream(self, question: str) -> tuple[int, str, dict]:
        """Legge solo il primo evento SSE e chiude la socket senza ``done``."""
        sock, status, content_type, _, event = self.open_stream_socket(question)
        try:
            return status, content_type, event
        finally:
            self.close_with_rst(sock)

    def _kill_tree(self) -> None:
        assert self.process is not None
        if self.process.poll() is not None:
            return
        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            except OSError:
                self.process.terminate()
        else:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            if self.process is not None:
                self._kill_tree()
                try:
                    self.process.wait(timeout=STOP_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        try:
                            subprocess.run(
                                ["taskkill", "/PID", str(self.process.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                check=False,
                            )
                        except OSError:
                            self.process.kill()
                    else:
                        try:
                            os.killpg(self.process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        try:
                            self.process.wait(timeout=STOP_TIMEOUT_SECONDS)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                            self.process.wait(timeout=STOP_TIMEOUT_SECONDS)

            deadline = time.monotonic() + STOP_TIMEOUT_SECONDS
            while time.monotonic() < deadline and port_is_listening(self.port):
                time.sleep(POLL_INTERVAL_SECONDS)
            if port_is_listening(self.port):
                raise AssertionError(
                    f"il server e' ancora in ascolto sulla porta {self.port} dopo lo stop.\n"
                    f"output:\n{self._output()}"
                )
        finally:
            self.log.close()


def test_server_e2e_extractive_without_api_key():
    cases = [
        (
            "Come si passa dal DNA alla proteina?",
            "bio.dna",
            ["trascrizione", "traduzione", "codone"],
        ),
        (
            "In che anno è ambientato il romanzo di Manzoni?",
            "cul.manzoni",
            ["1628"],
        ),
    ]
    rows = load_eval_rows()
    by_question = {row["question"]: row for row in rows}
    abstain_rows = [row for row in rows if row.get("should_abstain")]
    assert len(abstain_rows) >= 2, "il test richiede almeno due domande fuori ambito"

    with RunningServer() as server:
        for question, document_id, keywords in cases:
            row = by_question.get(question)
            assert row is not None, f"domanda assente dall'eval: {question}"
            assert row.get("expect_doc") == document_id, (question, row.get("expect_doc"))
            assert set(row.get("must_include") or []) == set(keywords), (question, row.get("must_include"))

            answer = server.ask(question)
            assert answer.get("backend") == "extractive", answer
            assert answer.get("degraded") is False, answer
            assert answer.get("abstained") is False, answer
            text = answer.get("text") or ""
            assert text, f"risposta vuota per {question}"

            normalized = normalize(text)
            for keyword in keywords:
                assert normalize(keyword) in normalized, (question, keyword, text)

            sources = answer.get("sources") or []
            document_ids = {source.get("document_id") for source in sources}
            assert document_id in document_ids, (question, document_ids)
            numbers = {
                int(number)
                for source in sources
                for number in (source.get("numbers") or [])
            }
            cited = {int(number) for number in re.findall(r"\[(\d+)\]", text)}
            assert cited, f"nessuna citazione per {question}: {text}"
            assert cited <= numbers, f"citazioni orfane per {question}: {sorted(cited - numbers)}"

        for row in abstain_rows:
            question = row["question"]
            answer = server.ask(question)
            assert answer.get("backend") == "extractive", answer
            assert answer.get("abstained") is True, (question, answer)
            assert not answer.get("sources"), (question, answer.get("sources"))
            assert not re.search(r"\[\d+\]", answer.get("text") or ""), (question, answer.get("text"))

    assert not port_is_listening(server.port), f"porta {server.port} ancora in ascolto"


def test_server_e2e_stream_is_incremental_and_grounded():
    """The SSE path must be a real event stream, not just an HTTP 200."""
    cases = [
        (
            "Come si passa dal DNA alla proteina?",
            "bio.dna",
            ["trascrizione", "traduzione", "codone"],
        ),
        (
            "In che anno è ambientato il romanzo di Manzoni?",
            "cul.manzoni",
            ["1628"],
        ),
    ]
    rows = load_eval_rows()
    by_question = {row["question"]: row for row in rows}
    abstain_rows = [row for row in rows if row.get("should_abstain")]
    assert len(abstain_rows) >= 2, "il test richiede almeno due domande fuori ambito"

    with RunningServer() as server:
        for question, document_id, keywords in cases:
            row = by_question.get(question)
            assert row is not None, f"domanda assente dall'eval: {question}"
            assert row.get("expect_doc") == document_id, (question, row.get("expect_doc"))
            assert set(row.get("must_include") or []) == set(keywords), (question, row.get("must_include"))

            status, content_type, cache_control, events = server.stream(question)
            answer = server.ask(question)
            assert status == 200
            assert content_type.lower().startswith("text/event-stream"), content_type
            assert "no-store" in (cache_control or "").lower(), cache_control
            assert len(events) >= 3, f"stream troppo corto per {question}: {events!r}"
            assert events[0].get("type") == "meta", events[0]
            assert events[-1].get("type") == "done", events[-1]
            assert all(event.get("type") != "done" for event in events[:-1]), events
            assert not any(event.get("type") == "degraded" for event in events), events

            meta, done = events[0], events[-1]
            assert meta.get("backend") == "extractive", meta
            assert done.get("backend") == "extractive", done
            assert meta.get("cached") is False, meta
            assert done.get("cached") is False, done
            assert meta.get("abstained") is False, meta
            assert done.get("abstained") is False, done
            pieces = [event.get("text", "") for event in events if event.get("type") == "token"]
            assert pieces, f"nessun token nello stream per {question}"
            stream_text = "".join(pieces)
            assert stream_text == answer.get("text"), (question, stream_text, answer.get("text"))

            normalized = normalize(stream_text)
            for keyword in keywords:
                assert normalize(keyword) in normalized, (question, keyword, stream_text)

            sources = meta.get("sources") or []
            document_ids = {source.get("document_id") for source in sources}
            assert document_id in document_ids, (question, document_ids)
            numbers = {
                int(number)
                for source in sources
                for number in (source.get("numbers") or [])
            }
            cited = {int(number) for number in re.findall(r"\[(\d+)\]", stream_text)}
            assert cited, f"nessuna citazione nello stream per {question}: {stream_text}"
            assert cited <= numbers, f"citazioni orfane nello stream per {question}: {sorted(cited - numbers)}"

        for row in abstain_rows:
            question = row["question"]
            status, content_type, _, events = server.stream(question)
            answer = server.ask(question)
            assert status == 200
            assert content_type.lower().startswith("text/event-stream"), content_type
            assert events[0].get("type") == "meta", events[0]
            assert events[-1].get("type") == "done", events[-1]
            assert len(events) >= 3, f"stream troppo corto per {question}: {events!r}"
            assert events[0].get("backend") == "extractive", events[0]
            assert events[-1].get("backend") == "extractive", events[-1]
            assert events[0].get("cached") is False, events[0]
            assert events[-1].get("cached") is False, events[-1]
            assert events[0].get("abstained") is True, events[0]
            assert events[-1].get("abstained") is True, events[-1]
            assert not events[0].get("sources"), events[0].get("sources")
            stream_text = "".join(
                event.get("text", "") for event in events if event.get("type") == "token"
            )
            assert stream_text == answer.get("text"), (question, stream_text, answer.get("text"))
            assert not re.search(r"\[\d+\]", stream_text), (question, stream_text)

    assert not port_is_listening(server.port), f"porta {server.port} ancora in ascolto"


def test_server_e2e_survives_early_stream_disconnect():
    question = "Come si passa dal DNA alla proteina?"
    row = {row["question"]: row for row in load_eval_rows()}.get(question)
    assert row is not None, f"domanda assente dall'eval: {question}"
    assert row.get("expect_doc") == "bio.dna", row
    keywords = row.get("must_include") or []
    assert set(keywords) == {"trascrizione", "traduzione", "codone"}, keywords

    with RunningServer(
        env_overrides={STREAM_TEST_DELAY_ENV: STREAM_TEST_DELAY_MS},
        instrumented=True,
    ) as server:
        status, content_type, first_event = server.disconnect_mid_stream(question)
        assert status == 200
        assert content_type.lower().startswith("text/event-stream"), content_type
        assert first_event.get("type") == "meta", first_event
        assert first_event.get("abstained") is False, first_event
        assert "bio.dna" in {
            source.get("document_id") for source in (first_event.get("sources") or [])
        }, first_event

        health = server.wait_for_stream_aborts()
        assert health.get("status") == "ok", health
        assert health.get("backend") == "extractive", health
        assert int(health.get("stream_aborts") or 0) >= 1, health

        status, content_type, _, events = server.stream(question)
        assert status == 200
        assert content_type.lower().startswith("text/event-stream"), content_type
        assert len(events) >= 3, events
        assert events[-1].get("type") == "done", events[-1]
        assert all(event.get("type") != "done" for event in events[:-1]), events
        stream_text = "".join(
            event.get("text", "") for event in events if event.get("type") == "token"
        )
        assert stream_text, "stream completo senza testo"
        for keyword in keywords:
            assert normalize(keyword) in normalize(stream_text), (keyword, stream_text)

        meta = events[0]
        sources = meta.get("sources") or []
        assert "bio.dna" in {source.get("document_id") for source in sources}, meta
        numbers = {
            int(number)
            for source in sources
            for number in (source.get("numbers") or [])
        }
        cited = {int(number) for number in re.findall(r"\[(\d+)\]", stream_text)}
        assert cited, stream_text
        assert cited <= numbers, sorted(cited - numbers)

        answer = server.ask(question)
        assert answer.get("backend") == "extractive", answer
        assert answer.get("abstained") is False, answer
        assert "bio.dna" in {
            source.get("document_id") for source in (answer.get("sources") or [])
        }, answer
        assert stream_text == answer.get("text"), (stream_text, answer.get("text"))

        log_text = server.output()
        assert "Traceback (most recent call last)" not in log_text, log_text[-2000:]
        assert "Exception occurred during processing" not in log_text, log_text[-2000:]

    assert server.process is not None, "processo del server assente"
    assert server.process.poll() is not None, "processo del server ancora vivo dopo il teardown"
    assert not port_is_listening(server.port), f"porta {server.port} ancora in ascolto"


def test_server_e2e_concurrent_aborts_and_complete_stream():
    """Cento RST e uno stream completo, tenuti aperti tutti insieme."""
    question = "Come si passa dal DNA alla proteina?"
    expected_aborts = CONCURRENT_ABORTS
    with RunningServer(
        env_overrides={
            STREAM_TEST_DELAY_ENV: CONCURRENT_STREAM_DELAY_MS,
            CONCURRENT_SWITCH_INTERVAL_ENV: CONCURRENT_SWITCH_INTERVAL,
            CONCURRENT_ABORT_YIELD_ENV: CONCURRENT_ABORT_YIELD_MS,
            CONCURRENT_ABORT_BARRIER_ENV: CONCURRENT_ABORT_BARRIER,
        },
        instrumented=True,
    ) as server:
        initial = server.health()
        assert int(initial.get("stream_aborts") or 0) == 0, initial

        # La cache viene riempita da ask; il baseline normale e' quindi cache
        # hit, con latency_ms fisso e confronto raw deterministico.
        answer = server.ask(question)
        assert answer.get("backend") == "extractive", answer
        base_status, base_type, base_cache, base_raw = server.stream_raw(question)
        assert base_status == 200
        assert base_type.lower().startswith("text/event-stream"), base_type
        assert "no-store" in (base_cache or "").lower(), base_cache
        assert b"stream_aborts" not in base_raw
        baseline_events = parse_sse_events(base_raw)
        assert baseline_events[0]["type"] == "meta", baseline_events[0]
        assert baseline_events[-1]["type"] == "done", baseline_events[-1]

        before = int(server.health().get("stream_aborts") or 0)
        abort_sockets: list[socket.socket] = []
        healthy_sock: socket.socket | None = None
        healthy_raw = b""
        try:
            # Ogni abort ha gia' ricevuto meta e resta aperto: il server ha
            # ancora token/done da scrivere. Il client sano viene aperto
            # mentre questi handler sono ancora vivi, poi tutti
            # vengono chiusi con RST nello stesso burst.
            for _ in range(expected_aborts):
                sock, abort_status, abort_type, _, first_event = server.open_stream_socket(question)
                abort_sockets.append(sock)
                assert abort_status == 200
                assert abort_type.lower().startswith("text/event-stream"), abort_type
                assert first_event.get("type") == "meta", first_event

            healthy_sock, healthy_status, healthy_type, healthy_first, healthy_first_event = (
                server.open_stream_socket(question)
            )
            assert healthy_status == 200
            assert healthy_type == base_type
            assert healthy_first_event.get("type") == "meta", healthy_first_event

            for sock in abort_sockets:
                server.close_with_rst(sock)
            abort_sockets.clear()

            healthy_raw = server.read_stream_rest(healthy_sock, healthy_first)
        finally:
            for sock in abort_sockets:
                try:
                    server.close_with_rst(sock)
                except OSError:
                    pass
            if healthy_sock is not None:
                healthy_sock.close()

        health = server.wait_for_stream_aborts(expected=expected_aborts)
        after = int(health.get("stream_aborts") or 0)
        delta = after - before
        print(f"  stream_aborts delta={delta} (atteso {expected_aborts})")
        assert delta == expected_aborts, (before, after, expected_aborts)
        assert health.get("status") == "ok", health
        assert health.get("backend") == "extractive", health

        assert healthy_raw, "client sano senza risposta"
        assert b"stream_aborts" not in healthy_raw
        assert healthy_raw == base_raw, (
            f"payload SSE diverso sotto concorrenza: "
            f"{len(healthy_raw)} byte vs {len(base_raw)} byte"
        )

        events = parse_sse_events(healthy_raw)
        assert events[-1].get("type") == "done", events[-1]
        text = "".join(
            event.get("text", "") for event in events if event.get("type") == "token"
        )
        for keyword in ("trascrizione", "traduzione", "codone"):
            assert normalize(keyword) in normalize(text), (keyword, text)
        meta = events[0]
        sources = meta.get("sources") or []
        assert "bio.dna" in {source.get("document_id") for source in sources}, meta
        numbers = {
            int(number)
            for source in sources
            for number in (source.get("numbers") or [])
        }
        cited = {int(number) for number in re.findall(r"\[(\d+)\]", text)}
        assert cited and cited <= numbers, (cited, numbers, text)

        log_text = server.output()
        assert "Traceback (most recent call last)" not in log_text, log_text[-2000:]
        assert "Exception occurred during processing" not in log_text, log_text[-2000:]

    assert server.process is not None, "processo del server assente"
    assert server.process.poll() is not None, "processo del server ancora vivo dopo il teardown"
    assert not port_is_listening(server.port), f"porta {server.port} ancora in ascolto"


def _run_all() -> int:
    from lligma.tutor.console import setup_utf8

    setup_utf8()
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} test superati")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
