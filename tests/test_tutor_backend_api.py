"""The API backend, tested against a real HTTP server speaking the OpenAI wire format.

This path cannot be checked against the real provider without a key, and an
untested HTTP client is where the failures hide: a stream parser that only works
when the provider sends one chunk, an error body swallowed instead of raised, a
missing key that turns into an empty answer instead of a clear message.

The mock below speaks the same protocol, chunking included, so the client code
is exercised end to end.

Run with pytest, or directly: python tests/test_tutor_backend_api.py
"""

from __future__ import annotations

import json
import os
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lligma.tutor.backends import AnswerRequest, BackendError, ExtractiveBackend, OpenAIBackend  # noqa: E402
from lligma.tutor.config import OpenAIConfig, TutorConfig  # noqa: E402
from lligma.tutor.index import BM25Index, Chunk, Hit  # noqa: E402
from lligma.tutor.service import TutorService  # noqa: E402

ANSWER_TEXT = "La clorofilla cattura la luce nella fase luminosa [1]."


class MockTeacher(BaseHTTPRequestHandler):
    status = 200
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # silenzioso nei test
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")

        if self.status != 200:
            body = json.dumps({"error": {"message": "boom"}}).encode()
            self.send_response(self.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if payload.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for piece in ("La clorofilla ", "cattura la luce ", "nella fase luminosa [1]."):
                event = {"choices": [{"delta": {"content": piece}}]}
                self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return

        body = json.dumps({"choices": [{"message": {"content": ANSWER_TEXT}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @classmethod
    def serve(cls):
        server = ThreadingHTTPServer(("127.0.0.1", 0), cls)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, f"http://127.0.0.1:{server.server_address[1]}"


def make_hit() -> Hit:
    chunk = Chunk(
        document_id="bio.fotosintesi",
        title="La fotosintesi clorofilliana",
        subject="biologia",
        source="Dispensa",
        text="La clorofilla cattura la luce. L'energia serve a scindere l'acqua.",
        ordinal=0,
        terms={},
        length=14,
    )
    return Hit(score=9.0, chunk=chunk)


def make_request() -> AnswerRequest:
    return AnswerRequest(question="Cosa fa la clorofilla?", hits=[make_hit()])


def backend_for(url: str, key: str = "test-key") -> OpenAIBackend:
    os.environ["TUTOR_TEST_KEY"] = key
    return OpenAIBackend(
        OpenAIConfig(base_url=url, model="mock-model", api_key_env="TUTOR_TEST_KEY", timeout=10)
    )


def test_backend_is_unavailable_without_key():
    os.environ["TUTOR_EMPTY_KEY"] = ""
    backend = OpenAIBackend(OpenAIConfig(base_url="http://127.0.0.1:9", api_key_env="TUTOR_EMPTY_KEY"))
    assert backend.available() is False
    try:
        backend.answer(make_request())
    except BackendError as error:
        assert "TUTOR_EMPTY_KEY" in str(error)
    else:
        raise AssertionError("senza chiave il backend deve rifiutare, non inventare")


def test_backend_returns_the_message_content():
    server, url = MockTeacher.serve()
    try:
        answer = backend_for(url).answer(make_request())
    finally:
        server.shutdown()
        server.server_close()
    assert answer == ANSWER_TEXT


def test_backend_streams_deltas_in_order():
    server, url = MockTeacher.serve()
    try:
        pieces = list(backend_for(url).stream(make_request()))
    finally:
        server.shutdown()
        server.server_close()
    assert "".join(pieces) == "La clorofilla cattura la luce nella fase luminosa [1]."
    assert len(pieces) == 3, "ogni delta deve diventare un pezzo, non uno solo"


def test_http_error_becomes_a_backend_error():
    class Broken(MockTeacher):
        status = 500

    server, url = Broken.serve()
    try:
        backend_for(url).answer(make_request())
    except BackendError as error:
        assert "500" in str(error)
    else:
        raise AssertionError("un 500 deve sollevare BackendError")
    finally:
        server.shutdown()
        server.server_close()


def test_unreachable_host_raises_instead_of_hanging():
    backend = backend_for("http://127.0.0.1:1")
    try:
        backend.answer(make_request())
    except BackendError as error:
        assert "non raggiungibile" in str(error).lower()
    else:
        raise AssertionError("un host irraggiungibile deve sollevare BackendError")


def test_service_uses_the_api_backend_when_configured():
    from build_tutor_index import collect

    server, url = MockTeacher.serve()
    try:
        config = TutorConfig.from_file(ROOT / "configs" / "tutor.yaml")
        config.generation.backend = "openai"
        config.openai = OpenAIConfig(
            base_url=url, model="mock-model", api_key_env="TUTOR_TEST_KEY", timeout=10
        )
        index = BM25Index.build(collect(ROOT / "data" / "tutor", "generale"))
        os.environ["TUTOR_TEST_KEY"] = "test-key"
        service = TutorService(config, index=index, backend=OpenAIBackend(config.openai))
        answer = service.ask("Cosa fa la clorofilla?", use_cache=False)
    finally:
        server.shutdown()
        server.server_close()
    assert answer.backend == "openai"
    assert answer.text == ANSWER_TEXT
    assert answer.sources, "anche una risposta generata deve mostrare le fonti"


def test_service_falls_back_when_the_api_is_down():
    from build_tutor_index import collect

    config = TutorConfig.from_file(ROOT / "configs" / "tutor.yaml")
    config.openai = OpenAIConfig(
        base_url="http://127.0.0.1:1", model="mock", api_key_env="TUTOR_TEST_KEY", timeout=2
    )
    os.environ["TUTOR_TEST_KEY"] = "test-key"
    index = BM25Index.build(collect(ROOT / "data" / "tutor", "generale"))
    service = TutorService(
        config, index=index, backend=OpenAIBackend(config.openai)
    )
    answer = service.ask("Quando si pota la vite?", use_cache=False)
    assert answer.degraded is True
    assert answer.backend == "extractive"
    assert "inverno" in answer.text.lower()
    assert isinstance(service.backend, OpenAIBackend)
    assert ExtractiveBackend


def test_dotenv_is_loaded_without_overriding_the_environment(tmp_path=None):
    from lligma.tutor.env import load_dotenv

    directory = ROOT / "outputs" / "test-env"
    directory.mkdir(parents=True, exist_ok=True)
    env_file = directory / ".env"
    env_file.write_text(
        "TUTOR_DOTENV_NEW=dal-file\nTUTOR_DOTENV_EXISTING=dal-file\n# commento\n",
        encoding="utf-8",
    )
    os.environ.pop("TUTOR_DOTENV_NEW", None)
    os.environ["TUTOR_DOTENV_EXISTING"] = "dall-ambiente"
    try:
        added = load_dotenv(env_file)
        assert added == 1
        assert os.environ["TUTOR_DOTENV_NEW"] == "dal-file"
        assert os.environ["TUTOR_DOTENV_EXISTING"] == "dall-ambiente", "l'ambiente vince sul file"
    finally:
        os.environ.pop("TUTOR_DOTENV_NEW", None)
        os.environ.pop("TUTOR_DOTENV_EXISTING", None)
        env_file.unlink(missing_ok=True)


def _run_all() -> int:
    from lligma.tutor.console import setup_utf8

    setup_utf8()
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except (AssertionError, BackendError) as error:
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} test superati")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
