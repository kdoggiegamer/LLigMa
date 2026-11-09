"""The HTTP layer: JSON answers, SSE streaming, and the path guard.

The server is small, but it is the only part a student's browser talks to, so
the failure modes that matter are tested directly: malformed JSON must be a 400
and not a crash, a request for ../../etc/passwd must be refused, an untrusted
Host must not reach the loopback service, a rejected body must close the
connection before it can be reused, and the SSE stream must actually deliver
events in order.

Run with pytest, or directly: python tests/test_tutor_server.py
"""

from __future__ import annotations

import http.client
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lligma.tutor.backends import ExtractiveBackend  # noqa: E402
from lligma.tutor.config import TutorConfig  # noqa: E402
from lligma.tutor.index import BM25Index  # noqa: E402
from lligma.tutor.server import TutorRequestHandler, build_server, write_sse_event  # noqa: E402
from lligma.tutor.service import TutorService  # noqa: E402


class RunningServer:
    def __init__(self):
        from build_tutor_index import collect

        config = TutorConfig.from_file(ROOT / "configs" / "tutor.yaml")
        config.generation.backend = "extractive"
        config.server.host = "127.0.0.1"
        config.server.port = 0
        index = BM25Index.build(collect(ROOT / "data" / "tutor", "generale"))
        service = TutorService(config, index=index, backend=ExtractiveBackend())
        self.server = build_server(config, service)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=10) as response:
            return response.status, response.read()

    def post(self, path: str, payload, raw: bytes | None = None, headers: dict | None = None):
        body = raw if raw is not None else json.dumps(payload).encode()
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers or {})
        request = urllib.request.Request(
            self.base + path, data=body,
            headers=request_headers, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    def raw_post(self, path: str, body: bytes, headers: dict | None = None):
        """POST senza seguire redirect, utile per Host e Connection."""
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            connection.request("POST", path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()


def test_health_reports_the_index():
    with RunningServer() as server:
        status, body = server.get("/api/health")
        payload = json.loads(body)
    assert status == 200
    assert payload["status"] == "ok"
    assert payload["chunks"] > 10
    assert "agronomia" in payload["subjects"]


def test_ask_returns_an_answer_with_sources():
    with RunningServer() as server:
        status, body = server.post("/api/ask", {"question": "Come si pota la vite?"})
        payload = json.loads(body)
    assert status == 200
    assert payload["text"]
    assert payload["sources"]
    assert payload["sources"][0]["title"]


def test_sources_list_each_sheet_once():
    """Il payload HTTP non deve ripetere la stessa scheda due volte."""
    with RunningServer() as server:
        status, body = server.post(
            "/api/ask", {"question": "Che cos'e' la fotosintesi clorofilliana?"}
        )
        payload = json.loads(body)
    assert status == 200
    titles = [source["title"] for source in payload["sources"]]
    assert titles, "nessuna fonte nella risposta"
    assert len(titles) == len(set(titles)), f"scheda ripetuta: {titles}"
    numbers = [number for source in payload["sources"] for number in source["numbers"]]
    assert numbers, "manca il riferimento ai passaggi usati"
    assert sorted(numbers) == sorted(set(numbers)), f"passaggio citato due volte: {numbers}"


def test_cross_origin_post_is_refused():
    """Una pagina di un altro sito non deve poter usare la chiave API."""
    with RunningServer() as server:
        status, body = server.post(
            "/api/ask",
            {"question": "Come si pota la vite?"},
            headers={"Origin": "http://evil.example"},
        )
    assert status == 403
    assert "origine" in json.loads(body)["error"]


def test_same_origin_post_is_allowed():
    with RunningServer() as server:
        status, body = server.post(
            "/api/ask",
            {"question": "Come si pota la vite?"},
            headers={"Origin": server.base},
        )
    assert status == 200
    assert json.loads(body)["text"]


def test_malformed_json_is_a_400_not_a_crash():
    with RunningServer() as server:
        status, body = server.post("/api/ask", None, raw=b"{non json")
    assert status == 400
    assert "error" in json.loads(body)


def test_missing_question_is_a_400():
    with RunningServer() as server:
        status, _ = server.post("/api/ask", {"subject": "agronomia"})
    assert status == 400


def test_host_header_is_limited_to_loopback_names():
    """Un host arbitrario non deve poter raggiungere il server loopback."""
    body = json.dumps({"question": "Come si pota la vite?"}).encode("utf-8")
    with RunningServer() as server:
        status, headers, _ = server.raw_post(
            "/api/ask",
            body,
            {"Content-Type": "application/json", "Host": "evil.example"},
        )
        assert status == 403, status
        assert headers.get("Connection", "").lower() == "close", headers

        status, _, payload = server.raw_post(
            "/api/ask",
            body,
            {"Content-Type": "application/json", "Host": f"localhost:{server.port}"},
        )
    assert status == 200, payload


def test_rejected_post_closes_before_unread_body_is_reused():
    """Un 403/413 non deve lasciare il corpo nella pipeline keep-alive."""
    body = json.dumps({"question": "Come si pota la vite?"}).encode("utf-8")
    with RunningServer() as server:
        status, headers, _ = server.raw_post(
            "/api/ask",
            body,
            {
                "Content-Type": "application/json",
                "Origin": "http://evil.example",
            },
        )
        assert status == 403, status
        assert headers.get("Connection", "").lower() == "close", headers

        status, headers, _ = server.raw_post(
            "/api/ask",
            body,
            {
                "Content-Type": "application/json",
                "Content-Length": str(64 * 1024 + 1),
            },
        )
    assert status == 413, status
    assert headers.get("Connection", "").lower() == "close", headers


def test_path_traversal_is_refused():
    with RunningServer() as server:
        for path in ("/../../configs/tutor.yaml", "/%2e%2e/%2e%2e/configs/tutor.yaml"):
            try:
                status, _ = server.get(path)
            except urllib.error.HTTPError as error:
                status = error.code
            assert status in {400, 403, 404}, path


def test_stream_delivers_ordered_events():
    with RunningServer() as server:
        status, body = server.post(
            "/api/stream", {"question": "Che cosa e' la fotosintesi?", "subject": "biologia"}
        )
    assert status == 200
    events = []
    for block in body.decode("utf-8").split("\n\n"):
        line = block.strip()
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    assert events, "lo stream non ha prodotto eventi"
    assert events[0]["type"] == "meta"
    assert events[-1]["type"] == "done"
    assert any(event["type"] == "token" for event in events)
    assert events[0]["sources"]


def test_stream_handler_suppresses_a_closed_client_socket():
    """Una socket chiusa non deve produrre un traceback ne' fermare il server."""

    class ClosedWriter:
        def write(self, chunk):
            raise BrokenPipeError("client chiuso")

        def flush(self):
            raise AssertionError("flush non deve essere raggiunto dopo una write fallita")

    class OneEventService:
        def stream(self, question, subject=None, history=None):
            yield {"type": "meta", "sources": [], "backend": "extractive", "abstained": False}
            yield {"type": "token", "text": "testo"}

    writer = ClosedWriter()
    assert write_sse_event(writer, {"type": "token", "text": "x"}) is False

    handler = object.__new__(TutorRequestHandler)
    handler.wfile = writer
    handler.service = OneEventService()
    handler.close_connection = False
    handler.send_response = lambda *args, **kwargs: None
    handler.send_header = lambda *args, **kwargs: None
    handler.end_headers = lambda *args, **kwargs: None
    handler._stream("domanda", None, [])
    assert handler.close_connection is True


def test_index_page_is_served():
    with RunningServer() as server:
        status, body = server.get("/")
    assert status == 200
    assert b"LLigMa Tutor" in body
    assert "text/html" in "text/html"


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
