"""Server HTTP del tutor: solo libreria standard.

Perche' non FastAPI: il servizio ha due endpoint, un file statico e uno stream
SSE. ``http.server`` li copre senza installare nulla, il che conta quando il
destinatario e' un laboratorio scolastico con un Python qualsiasi. Se in futuro
servono autenticazione, quota per studente e log strutturati, il posto giusto e'
un reverse proxy davanti, non dentro questo file.

Sicurezza di base, deliberata:
* bind su 127.0.0.1 per default;
* body massimo 64 KB e domanda massima 800 caratteri;
* nessun percorso fuori da ``web/`` (controllo con realpath);
* header CSP e ``nosniff`` su ogni risposta.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from .config import TutorConfig
from .service import TutorService

MAX_BODY_BYTES = 64 * 1024
MAX_QUESTION_CHARS = 800
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
    ".woff2": "font/woff2",
}

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; img-src 'self' data:; "
        "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
        "form-action 'self'"
    ),
    "X-Frame-Options": "DENY",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

CONNECTION_LOSS_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


def write_sse_event(wfile, event: dict) -> bool:
    """Scrive un evento SSE e sopprime solo la chiusura del client.

    Restituisce ``False`` quando la connessione e' gia' morta, cosi' il
    chiamante smette di generare e di scrivere altri eventi.
    """
    chunk = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
    try:
        wfile.write(chunk)
        wfile.flush()
    except CONNECTION_LOSS_ERRORS:
        return False
    return True


class TutorRequestHandler(BaseHTTPRequestHandler):
    service: TutorService
    web_dir: Path
    protocol_version = "HTTP/1.1"
    server_version = "LLigMaTutor/1.0"

    # -- infrastruttura ------------------------------------------------------

    def log_message(self, fmt: str, *args) -> None:  # noqa: D102 - firma stdlib
        sys.stderr.write(f"[tutor] {self.address_string()} {fmt % args}\n")

    def _send(self, status: int, body: bytes, content_type: str, extra: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)
        for key, value in (extra or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: dict, extra: dict | None = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Cache-Control": "no-store"}
        headers.update(extra or {})
        self._send(status, body, "application/json; charset=utf-8", headers)

    def _error(self, status: int, message: str, close: bool = False) -> None:
        if close:
            # Se il corpo non e' stato letto, riusare questa connessione
            # farebbe interpretare i byte residui come una nuova richiesta.
            self.close_connection = True
        self._json(
            status,
            {"error": message},
            {"Connection": "close"} if close else None,
        )

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "Content-Length non valido", close=True)
            return None
        if length <= 0:
            self._error(HTTPStatus.BAD_REQUEST, "corpo della richiesta mancante", close=True)
            return None
        if length > MAX_BODY_BYTES:
            self._error(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "richiesta troppo grande",
                close=True,
            )
            return None
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(HTTPStatus.BAD_REQUEST, "JSON non valido")
            return None
        if not isinstance(payload, dict):
            self._error(HTTPStatus.BAD_REQUEST, "atteso un oggetto JSON")
            return None
        return payload

    @staticmethod
    def _clean_question(payload: dict) -> str | None:
        question = payload.get("question") or payload.get("q")
        if not isinstance(question, str):
            return None
        question = question.strip()
        return question[:MAX_QUESTION_CHARS] or None

    @staticmethod
    def _clean_subject(payload: dict) -> str | None:
        subject = payload.get("subject")
        if not isinstance(subject, str):
            return None
        subject = subject.strip().lower()
        return subject or None

    def _host_allowed(self) -> bool:
        """Blocca il DNS rebinding sul bind loopback.

        Il confronto Origin/Host da solo non basta: una pagina su evil.example
        puo' far risolvere evil.example su 127.0.0.1, e allora Host e Origin
        coincidono. Con un bind loopback accettiamo solo i nomi locali, salvo
        una lista esplicita in configurazione per un reverse proxy.
        """
        allowed = getattr(self.server, "allowed_hosts", frozenset())
        if not allowed:
            return True
        host_header = (self.headers.get("Host") or "").strip()
        if not host_header:
            return False
        try:
            hostname = urlparse(f"//{host_header}").hostname
        except ValueError:
            return False
        return bool(hostname) and hostname.lower() in allowed

    def _origin_allowed(self) -> bool:
        """Rifiuta i POST inviati da pagine di altri siti.

        Senza questo controllo una pagina qualsiasi aperta nel browser dello
        studente puo' chiamare il tutor su localhost e consumare la chiave API.
        I client non browser (curl, script, test) non mandano ``Origin`` e
        restano ammessi; il confronto e' sull'host, non sullo schema, cosi'
        funziona anche dietro un reverse proxy TLS.
        """
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = (self.headers.get("Host") or "").strip().lower()
        try:
            origin_host = urlparse(origin).netloc.strip().lower()
        except ValueError:
            return False
        return bool(host) and bool(origin_host) and origin_host == host

    @staticmethod
    def _clean_history(payload: dict) -> list[dict]:
        history = payload.get("history")
        if not isinstance(history, list):
            return []
        cleaned = []
        for turn in history[-6:]:
            if isinstance(turn, dict) and isinstance(turn.get("content"), str):
                cleaned.append(
                    {
                        "role": turn.get("role") if turn.get("role") in {"user", "assistant"} else "user",
                        "content": turn["content"][:800],
                    }
                )
        return cleaned

    # -- GET -----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - nome imposto da stdlib
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "host non consentito", close=True)
            return

        path = unquote(urlparse(self.path).path)

        if path in {"/api/health", "/health"}:
            payload = self.service.health()
            # Diagnostica additiva: non entra negli eventi SSE e non cambia
            # la risposta osservata da un client normale.
            lock = getattr(self.server, "stream_aborts_lock", None)
            if lock is None:
                aborts = int(getattr(self.server, "stream_aborts", 0))
            else:
                with lock:
                    aborts = int(getattr(self.server, "stream_aborts", 0))
            payload["stream_aborts"] = aborts
            self._json(HTTPStatus.OK, payload)
            return

        if path in {"/", "/index.html"}:
            path = "/index.html"

        relative = path.lstrip("/")
        candidate = (self.web_dir / relative).resolve()
        try:
            candidate.relative_to(self.web_dir.resolve())
        except ValueError:
            self._error(HTTPStatus.FORBIDDEN, "percorso non consentito")
            return

        if not candidate.is_file():
            self._error(HTTPStatus.NOT_FOUND, "risorsa non trovata")
            return

        content_type = CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream")
        self._send(HTTPStatus.OK, candidate.read_bytes(), content_type)

    # -- POST ----------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802 - nome imposto da stdlib
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "host non consentito", close=True)
            return

        path = urlparse(self.path).path
        if path not in {"/api/ask", "/api/stream"}:
            self._error(HTTPStatus.NOT_FOUND, "endpoint non trovato", close=True)
            return

        if not self._origin_allowed():
            self._error(HTTPStatus.FORBIDDEN, "origine non consentita", close=True)
            return

        payload = self._read_json()
        if payload is None:
            return

        question = self._clean_question(payload)
        if question is None:
            self._error(HTTPStatus.BAD_REQUEST, "serve il campo 'question'")
            return

        subject = self._clean_subject(payload)
        history = self._clean_history(payload)

        if path == "/api/ask":
            answer = self.service.ask(question, subject=subject, history=history)
            self._json(HTTPStatus.OK, answer.to_dict())
            return

        self._stream(question, subject, history)

    def _note_stream_abort(self) -> None:
        server = getattr(self, "server", None)
        if server is None:
            return
        with server.stream_aborts_lock:
            server.stream_aborts = getattr(server, "stream_aborts", 0) + 1

    def _stream(self, question: str, subject: str | None, history: list[dict]) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()

        try:
            for event in self.service.stream(question, subject=subject, history=history):
                if not write_sse_event(self.wfile, event):
                    # Lo studente ha chiuso la scheda: non e' un errore del server.
                    self._note_stream_abort()
                    return
        except CONNECTION_LOSS_ERRORS:
            # Chiusura avvenuta tra due eventi, prima del prossimo write.
            self._note_stream_abort()
            return
        except Exception as error:  # pragma: no cover - rete reale
            try:
                message = {"type": "error", "message": f"errore interno: {error}"}
                self.wfile.write(f"data: {json.dumps(message, ensure_ascii=False)}\n\n".encode())
                self.wfile.flush()
            except OSError:
                pass
        finally:
            self.close_connection = True


def _allowed_hosts(config: TutorConfig) -> frozenset[str]:
    configured = config.server.allowed_hosts or []
    if isinstance(configured, str):
        configured = [configured]
    hosts = {str(host).strip().lower() for host in configured if str(host).strip()}
    if hosts:
        return frozenset(hosts)
    if (config.server.host or "").strip().lower() in LOOPBACK_HOSTS:
        return LOOPBACK_HOSTS
    # Un bind esplicito su LAN/0.0.0.0 resta invariato finche' l'operatore non
    # configura allowed_hosts: il tutor non deve rompersi in una rete didattica.
    return frozenset()


def build_server(config: TutorConfig, service: TutorService | None = None) -> ThreadingHTTPServer:
    """Server pronto ma non in ascolto: i test lo avviano su una porta effimera."""
    service = service or TutorService(config)
    handler = type("BoundTutorHandler", (TutorRequestHandler,), {
        "service": service,
        "web_dir": config.server.web_dir,
    })
    server = ThreadingHTTPServer((config.server.host, config.server.port), handler)
    server.daemon_threads = True
    server.stream_aborts = 0
    server.stream_aborts_lock = threading.Lock()
    server.allowed_hosts = _allowed_hosts(config)
    return server


def main(argv: list[str] | None = None) -> int:
    from .console import setup_utf8

    setup_utf8()
    parser = argparse.ArgumentParser(description="Server di LLigMa Tutor")
    parser.add_argument("--config", default="configs/tutor.yaml")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--backend", default=None, help="auto | extractive | openai | local")
    args = parser.parse_args(argv)

    config = TutorConfig.from_file(args.config)
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.backend:
        config.generation.backend = args.backend

    service = TutorService(config)
    server = build_server(config, service)

    print(f"LLigMa Tutor in ascolto su http://{config.server.host}:{server.server_address[1]}/")
    print(f"  indice : {len(service.index)} passaggi da {service.index.metadata.get('documents')} schede")
    print(f"  backend: {service.backend.name}")
    for note in service.startup_notes:
        print(f"  nota   : {note}")
    print("  Ctrl+C per fermare.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nArresto.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
