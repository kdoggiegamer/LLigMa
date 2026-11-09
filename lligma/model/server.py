"""Alessandria: playground web locale per un checkpoint LLigMa.

Il server carica il checkpoint con le classi del repository, espone una pagina
statica e uno stream SSE. Non e' un servizio pubblico: bind su loopback, body
limitato e una sola generazione alla volta, perche' il modello non e' pensato
per essere richiamato in concorrenza.

Metriche misurate qui, non stimate dal client:

* tempo totale dalla richiesta all'ultimo token;
* tempo al primo token (TTFT);
* token generati, contati dal ``TextStreamer``;
* token/s, token generati diviso tempo totale.

Uso:
    .venv/Scripts/python.exe -m lligma.model.server \
        --model outputs/lligma-small-smoke --port 8081
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote, urlparse

from lligma.tutor.backends import AnswerRequest, ExtractiveBackend
from lligma.tutor.config import TutorConfig
from lligma.tutor.prompt import (
    ABSTENTION_TEXT,
    format_context,
    is_abstention_text,
    sources_from_hits,
)
from lligma.tutor.service import TutorService

MAX_BODY_BYTES = 64 * 1024
MAX_PROMPT_CHARS = 4000
MAX_NEW_TOKENS = 256
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".ico": "image/x-icon",
    ".json": "application/json; charset=utf-8",
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


class _StopOnEvent:
    """StoppingCriteria minimale: ferma generate() quando il client chiude."""

    def __init__(self, event: threading.Event) -> None:
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001
        return self.event.is_set()


class _QueueStreamer:
    """TextStreamer che pubblica pezzi su una coda e conta i token veri.

    ``TextStreamer`` riceve i token ID da ``generate``: e' l'unico punto in cui
    possiamo contare i token generati senza ricostruirli dal testo decodificato.
    Il primo ``put`` contiene il prompt e viene lasciato passare senza contarlo.
    """

    def __init__(self, tokenizer, events: queue.Queue) -> None:
        from transformers import TextStreamer

        # Composizione invece di ereditarieta': l'API di TextStreamer cambia
        # facilmente tra versioni, ma put/end/on_finalized_text sono stabili.
        self._base = TextStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        self._events = events
        self.generated_tokens = 0
        # TextStreamer chiama on_finalized_text su se stesso; lo sostituiamo
        # con il nostro callback, che pubblica il testo sulla coda.
        self._base.on_finalized_text = self._on_text
        self._next_is_prompt = True

    def _on_text(self, text: str, stream_end: bool = False) -> None:
        if text:
            self._events.put(("token", text))
        if stream_end:
            self._events.put(("end", None))

    def put(self, value) -> None:  # noqa: ANN001
        if self._next_is_prompt:
            self._next_is_prompt = False
        else:
            values = value.tolist() if hasattr(value, "tolist") else value
            if isinstance(values, list):
                if values and isinstance(values[0], list):
                    self.generated_tokens += sum(len(row) for row in values)
                else:
                    self.generated_tokens += len(values)
            else:
                self.generated_tokens += 1
        self._base.put(value)

    def end(self) -> None:
        self._base.end()


@dataclass
class TutorPreparation:
    """Contesto e baseline pronti prima di chiamare il modello."""

    prompt: str
    sources: list[dict]
    baseline: str
    retrieval_ms: float
    baseline_ms: float
    abstained: bool
    retrieved: int
    subject: str | None
    context_chars: int


class TutorBridge:
    """Recupera dal Tutor e prepara il contesto per il rewriter LLigMa.

    Il Tutor extractive resta la fonte di verita': il modello riceve solo i
    passaggi recuperati e la sua risposta viene mostrata accanto alla baseline.
    Non si usa il backend ``local`` del Tutor: qui il modello e' LLigMa.
    """

    def __init__(self, config_path: str | Path) -> None:
        self.config = TutorConfig.from_file(config_path)
        self.backend = ExtractiveBackend(
            max_sentences=self.config.generation.max_sentences,
            min_overlap=self.config.generation.min_sentence_overlap,
        )
        self.service = TutorService(self.config, backend=self.backend)

    def health(self) -> dict:
        return {
            "chunks": len(self.service.index),
            "documents": self.service.index.metadata.get("documents"),
            "backend": "extractive",
        }

    def prepare(self, question: str) -> TutorPreparation:
        question = question.strip()[:800]
        started = time.perf_counter()
        hits, subject = self.service.retrieve(question)
        context, used = format_context(
            hits, max_chars=self.config.generation.max_context_chars
        )
        retrieval_ms = round((time.perf_counter() - started) * 1000, 1)

        if not used:
            return TutorPreparation(
                prompt="",
                sources=[],
                baseline=ABSTENTION_TEXT,
                retrieval_ms=retrieval_ms,
                baseline_ms=0.0,
                abstained=True,
                retrieved=0,
                subject=subject,
                context_chars=0,
            )

        baseline_started = time.perf_counter()
        request = AnswerRequest(
            question=question,
            hits=used,
            max_context_chars=self.config.generation.max_context_chars,
            max_sentences=self.config.generation.max_sentences,
            min_sentence_overlap=self.config.generation.min_sentence_overlap,
            match_query=question,
        )
        baseline = self.backend.answer(request).strip() or ABSTENTION_TEXT
        baseline_ms = round((time.perf_counter() - baseline_started) * 1000, 1)
        abstained = is_abstention_text(baseline)
        sources = [] if abstained else [
            source.to_dict() for source in sources_from_hits(used)
        ]

        prompt = (
            "Sei un assistente didattico italiano. Usa solo il CONTESTO fornito. "
            "Rispondi in italiano, in modo breve, citando le fonti con [n].\n\n"
            f"CONTESTO:\n{context}\n\n"
            f"DOMANDA: {question}\n\n"
            "RISPOSTA:"
        )
        return TutorPreparation(
            prompt=prompt,
            sources=sources,
            baseline=baseline,
            retrieval_ms=retrieval_ms,
            baseline_ms=baseline_ms,
            abstained=abstained,
            retrieved=len(used),
            subject=subject,
            context_chars=len(context),
        )


class ModelRuntime:
    """Caricamento e generazione serializzata di un checkpoint LLigMa."""

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        import torch
        from safetensors import safe_open
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        from .config import LLigMaConfig
        from .lligma import LLigMaForCausalLM

        self.path = Path(model_path).resolve()
        if not self.path.is_dir():
            raise FileNotFoundError(f"checkpoint non trovato: {self.path}")
        self.torch = torch
        self.device = self._resolve_device(device)
        self.lock = threading.Lock()

        values = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        for key in ("architectures", "dtype", "transformers_version"):
            values.pop(key, None)
        config = LLigMaConfig(**values)

        # Il checkpoint smoke e' stato salvato con embedding legati ma senza
        # il campo tie_word_embeddings nel config. Se manca lm_head.weight,
        # il tying va dedotto dal checkpoint: altrimenti il caricamento
        # standard crea una testa di output casuale.
        with safe_open(str(self.path / "model.safetensors"), framework="pt", device="cpu") as handle:
            checkpoint_keys = set(handle.keys())
        inferred_tie = (
            "lm_head.weight" not in checkpoint_keys
            and "model.embed_tokens.weight" in checkpoint_keys
        )
        self.tied = bool(getattr(config, "tie_word_embeddings", False) or inferred_tie)
        config.tie_word_embeddings = self.tied
        config.use_cache = True

        self.model = LLigMaForCausalLM(config)
        state = load_file(str(self.path / "model.safetensors"))
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # Con il tying il head non e' serializzato: e' un missing atteso, non
        # un peso da inizializzare a caso. Lo togliamo dal report operativo.
        self.missing = [key for key in missing if key != "lm_head.weight"]
        self.unexpected = list(unexpected)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.max_context = int(getattr(config, "max_position_embeddings", 8192))
        self.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _resolve_device(device: str) -> str:
        import torch

        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    def health(self) -> dict:
        return {
            "status": "ok",
            "model": str(self.path),
            "device": self.device,
            "parameters": self.parameters,
            "tied": self.tied,
            "missing": self.missing,
            "unexpected": self.unexpected,
            "max_context": self.max_context,
            "loaded_at": self.loaded_at,
        }

    def stream(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        started_at: float | None = None,
    ) -> Iterator[tuple[str, dict]]:
        """Genera e produce eventi (meta, token, done).

        Il generatore e' chiuso dal chiamante quando il client si disconnette:
        il ``finally`` alza lo stop e ferma ``generate`` al token successivo.
        """
        from transformers import StoppingCriteriaList

        torch = self.torch
        inputs = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        )
        input_ids = inputs["input_ids"].to(self.device)
        prompt_tokens = int(input_ids.shape[1])
        if prompt_tokens + max_new_tokens > self.max_context:
            raise ValueError(
                f"prompt troppo lungo: {prompt_tokens} token + "
                f"{max_new_tokens} nuovi superano il contesto di {self.max_context}"
            )

        events: queue.Queue = queue.Queue()
        streamer = _QueueStreamer(self.tokenizer, events)
        stop_event = threading.Event()
        errors: list[BaseException] = []
        started = started_at if started_at is not None else time.perf_counter()

        def run() -> None:
            try:
                with self.lock:
                    with torch.no_grad():
                        self.model.generate(
                            input_ids=input_ids,
                            attention_mask=torch.ones_like(input_ids),
                            max_new_tokens=max_new_tokens,
                            do_sample=temperature > 0,
                            temperature=max(temperature, 1e-5),
                            top_p=top_p,
                            repetition_penalty=repetition_penalty,
                            use_cache=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                            streamer=streamer,
                            stopping_criteria=StoppingCriteriaList(
                                [_StopOnEvent(stop_event)]
                            ),
                        )
            except BaseException as error:  # noqa: BLE001 - inoltrato al client
                errors.append(error)
                streamer.end()

        thread = threading.Thread(target=run, name="lligma-generate", daemon=True)
        thread.start()
        yield (
            "meta",
            {
                "model": self.path.name,
                "device": self.device,
                "parameters": self.parameters,
                "tied": self.tied,
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
            },
        )

        first_at: float | None = None
        pieces: list[str] = []
        try:
            while True:
                kind, payload = events.get()
                if kind == "token":
                    if first_at is None:
                        first_at = time.perf_counter()
                    pieces.append(payload)
                    yield (
                        "token",
                        {
                            "text": payload,
                            "tokens": streamer.generated_tokens,
                            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                        },
                    )
                elif kind == "end":
                    break
                elif kind == "error":
                    raise payload
        finally:
            stop_event.set()
            thread.join(timeout=5)

        if errors:
            raise errors[0]

        elapsed = time.perf_counter() - started
        text = "".join(pieces)
        tokens = streamer.generated_tokens
        yield (
            "done",
            {
                "text": text,
                "tokens": tokens,
                "elapsed_ms": round(elapsed * 1000, 1),
                "ttft_ms": round((first_at - started) * 1000, 1) if first_at else None,
                "tps": round(tokens / elapsed, 2) if elapsed > 0 else 0.0,
            },
        )


def _write_sse(wfile, event: dict) -> bool:
    chunk = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode("utf-8")
    try:
        wfile.write(chunk)
        wfile.flush()
    except CONNECTION_LOSS_ERRORS:
        return False
    return True


class PlaygroundHandler(BaseHTTPRequestHandler):
    runtime: ModelRuntime
    tutor: TutorBridge
    web_dir: Path
    protocol_version = "HTTP/1.1"
    server_version = "Alessandria/1.0"

    def log_message(self, fmt: str, *args) -> None:  # noqa: D102
        sys.stderr.write(f"[alessandria] {self.address_string()} {fmt % args}\n")

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: dict) -> None:
        self._send(
            status,
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
        )

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(HTTPStatus.BAD_REQUEST, "Content-Length non valido")
            return None
        if length <= 0:
            self._error(HTTPStatus.BAD_REQUEST, "corpo della richiesta mancante")
            return None
        if length > MAX_BODY_BYTES:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "richiesta troppo grande")
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

    def _host_allowed(self) -> bool:
        host = (self.headers.get("Host") or "").strip()
        if not host:
            return False
        try:
            hostname = urlparse(f"//{host}").hostname
        except ValueError:
            return False
        return bool(hostname) and hostname.lower() in LOOPBACK_HOSTS

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True
        host = (self.headers.get("Host") or "").strip().lower()
        try:
            origin_host = urlparse(origin).netloc.strip().lower()
        except ValueError:
            return False
        return bool(host) and origin_host == host

    @staticmethod
    def _clean_prompt(payload: dict) -> str | None:
        prompt = payload.get("prompt")
        if not isinstance(prompt, str):
            return None
        prompt = prompt.strip()
        return prompt[:MAX_PROMPT_CHARS] or None

    @staticmethod
    def _clean_question(payload: dict) -> str | None:
        question = payload.get("question") or payload.get("prompt")
        if not isinstance(question, str):
            return None
        question = question.strip()
        return question[:800] or None

    @staticmethod
    def _number(payload: dict, key: str, default, low, high):
        value = payload.get(key, default)
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} non numerico")
        if not low <= value <= high:
            raise ValueError(f"{key} fuori intervallo [{low}, {high}]")
        return value

    def _parameters(self, payload: dict) -> tuple[int, float, float, float]:
        max_new = self._number(payload, "max_new_tokens", 64, 1, MAX_NEW_TOKENS)
        if not max_new.is_integer():
            raise ValueError("max_new_tokens deve essere intero")
        temperature = self._number(payload, "temperature", 0.8, 0.0, 2.0)
        top_p = self._number(payload, "top_p", 0.9, 0.1, 1.0)
        repetition = self._number(payload, "repetition_penalty", 1.1, 1.0, 2.0)
        return int(max_new), temperature, top_p, repetition

    # -- GET -----------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "host non consentito")
            return
        path = unquote(urlparse(self.path).path)
        if path in {"/api/health", "/health"}:
            payload = self.runtime.health()
            payload["tutor"] = self.tutor.health()
            self._json(HTTPStatus.OK, payload)
            return
        if path in {"/", "/index.html"}:
            path = "/index.html"
        candidate = (self.web_dir / path.lstrip("/")).resolve()
        try:
            candidate.relative_to(self.web_dir.resolve())
        except ValueError:
            self._error(HTTPStatus.FORBIDDEN, "percorso non consentito")
            return
        if not candidate.is_file():
            self._error(HTTPStatus.NOT_FOUND, "risorsa non trovata")
            return
        self._send(
            HTTPStatus.OK,
            candidate.read_bytes(),
            CONTENT_TYPES.get(candidate.suffix.lower(), "application/octet-stream"),
        )

    # -- POST ----------------------------------------------------------------

    def do_POST(self) -> None:  # noqa: N802
        if not self._host_allowed():
            self._error(HTTPStatus.FORBIDDEN, "host non consentito")
            return
        if not self._origin_allowed():
            self._error(HTTPStatus.FORBIDDEN, "origine non consentita")
            return
        path = urlparse(self.path).path
        if path not in {"/api/stream", "/api/generate", "/api/tutor/stream"}:
            self._error(HTTPStatus.NOT_FOUND, "endpoint non trovato")
            return
        payload = self._read_json()
        if payload is None:
            return
        is_tutor = path == "/api/tutor/stream"
        prompt = self._clean_question(payload) if is_tutor else self._clean_prompt(payload)
        if prompt is None:
            self._error(HTTPStatus.BAD_REQUEST, "serve il campo 'question'" if is_tutor else "serve il campo 'prompt'")
            return
        try:
            max_new, temperature, top_p, repetition = self._parameters(payload)
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        if is_tutor:
            self._tutor_stream(prompt, max_new, temperature, top_p, repetition)
        elif path == "/api/generate":
            self._generate(prompt, max_new, temperature, top_p, repetition)
        else:
            self._stream(prompt, max_new, temperature, top_p, repetition)

    def _generate(
        self,
        prompt: str,
        max_new: int,
        temperature: float,
        top_p: float,
        repetition: float,
    ) -> None:
        events = self.runtime.stream(prompt, max_new, temperature, top_p, repetition)
        try:
            for kind, payload in events:
                if kind == "done":
                    self._json(HTTPStatus.OK, {"type": "done", **payload})
                    return
        except Exception as error:  # noqa: BLE001
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
        finally:
            events.close()

    def _start_sse(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)
        self.end_headers()

    def _stream(
        self,
        prompt: str,
        max_new: int,
        temperature: float,
        top_p: float,
        repetition: float,
    ) -> None:
        self._start_sse()

        events = self.runtime.stream(prompt, max_new, temperature, top_p, repetition)
        disconnected = False
        try:
            for kind, payload in events:
                if kind == "meta":
                    payload.setdefault("mode", "raw")
                    payload.setdefault("sources", [])
                event = {"type": kind, **payload}
                if not _write_sse(self.wfile, event):
                    disconnected = True
                    break
        except Exception as error:  # noqa: BLE001
            if not disconnected:
                _write_sse(self.wfile, {"type": "error", "message": str(error)})
        finally:
            events.close()
            self.close_connection = True

    def _tutor_stream(
        self,
        question: str,
        max_new: int,
        temperature: float,
        top_p: float,
        repetition: float,
    ) -> None:
        started = time.perf_counter()
        try:
            prepared = self.tutor.prepare(question)
        except Exception as error:  # noqa: BLE001
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(error))
            return

        self._start_sse()
        if prepared.abstained:
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            _write_sse(self.wfile, {
                "type": "meta",
                "mode": "tutor",
                "model": self.runtime.path.name,
                "device": self.runtime.device,
                "parameters": self.runtime.parameters,
                "tied": self.runtime.tied,
                "prompt_tokens": 0,
                "max_new_tokens": max_new,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition,
                "sources": [],
                "baseline": prepared.baseline,
                "retrieval_ms": prepared.retrieval_ms,
                "baseline_ms": prepared.baseline_ms,
                "retrieved": prepared.retrieved,
                "subject": prepared.subject,
                "context_chars": prepared.context_chars,
                "abstained": True,
            })
            _write_sse(self.wfile, {
                "type": "done",
                "text": "",
                "tokens": 0,
                "elapsed_ms": elapsed,
                "ttft_ms": elapsed,
                "tps": 0.0,
                "abstained": True,
            })
            self.close_connection = True
            return

        events = self.runtime.stream(
            prepared.prompt,
            max_new,
            temperature,
            top_p,
            repetition,
            started_at=started,
        )
        disconnected = False
        try:
            for kind, payload in events:
                if kind == "meta":
                    payload.update({
                        "mode": "tutor",
                        "sources": prepared.sources,
                        "baseline": prepared.baseline,
                        "retrieval_ms": prepared.retrieval_ms,
                        "baseline_ms": prepared.baseline_ms,
                        "retrieved": prepared.retrieved,
                        "subject": prepared.subject,
                        "context_chars": prepared.context_chars,
                        "abstained": False,
                    })
                event = {"type": kind, **payload}
                if not _write_sse(self.wfile, event):
                    disconnected = True
                    break
        except Exception as error:  # noqa: BLE001
            if not disconnected:
                _write_sse(self.wfile, {"type": "error", "message": str(error)})
        finally:
            events.close()
            self.close_connection = True


def build_server(
    runtime: ModelRuntime, tutor: TutorBridge, host: str, port: int
) -> ThreadingHTTPServer:
    web_dir = Path(__file__).resolve().parents[2] / "web" / "model"
    handler = type("BoundPlaygroundHandler", (PlaygroundHandler,), {
        "runtime": runtime,
        "tutor": tutor,
        "web_dir": web_dir,
    })
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alessandria: playground web per LLigMa")
    parser.add_argument("--model", default="outputs/lligma-small-smoke")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--device", default="cpu", help="cpu | cuda | auto")
    parser.add_argument("--tutor-config", default="configs/tutor.yaml")
    args = parser.parse_args(argv)

    runtime = ModelRuntime(args.model, args.device)
    tutor = TutorBridge(args.tutor_config)
    server = build_server(runtime, tutor, args.host, args.port)
    print(f"Alessandria su http://{args.host}:{server.server_address[1]}/")
    print(f"  modello : {runtime.path}")
    print(f"  device  : {runtime.device}")
    print(f"  parametri: {runtime.parameters:,}")
    print(f"  tying   : {runtime.tied}")
    print(f"  tutor   : {len(tutor.service.index)} passaggi da {tutor.service.index.metadata.get('documents')} schede")
    if runtime.missing or runtime.unexpected:
        print(f"  missing : {runtime.missing}")
        print(f"  unexpected: {runtime.unexpected}")
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
