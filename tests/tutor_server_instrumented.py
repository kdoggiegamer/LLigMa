"""Test-only server entrypoint: instrumentazione fuori dal codice di produzione.

Il server vero non conosce questo file. Qui, prima di avviare ``main()``, si
applicano al processo di test:

* una ``threading.Barrier`` che allinea gli handler degli abort;
* un trace hook che aggiunge un piccolo delay subito prima dello
  ``STORE_ATTR`` di ``_note_stream_abort``, rendendo osservabile una race sul
  contatore;
* un wrapper su ``write_sse_event`` per il delay SSE usato dai test di
  disconnessione;
* l'eventuale switch interval richiesto dal test;
* la cattura del solo ``ConnectionResetError`` atteso quando il client chiude
  con RST la connessione del test di disconnessione anticipata.

L'entrypoint non modifica il payload SSE: il wrapper scrive lo stesso evento e
restituisce lo stesso booleano, limitandosi a dormire dopo la scrittura. La
strumentazione è attiva solo se le relative variabili d'ambiente sono presenti.
Il ``ConnectionResetError`` è consumato esclusivamente nell'override di
``handle_one_request`` e solo per il rumore di RST del client; ogni altra
eccezione, compresi errori applicativi e traceback inattesi, continua a
propagarsi e a far fallire il test.
"""

from __future__ import annotations

import dis
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lligma.tutor import server as server_module  # noqa: E402
from lligma.tutor.server import TutorRequestHandler  # noqa: E402


def _seconds_from_env(name: str, maximum: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return 0.0
    try:
        return max(0.0, min(float(raw), maximum)) / 1000.0
    except ValueError:
        return 0.0


def _barrier_from_env() -> threading.Barrier | None:
    raw = os.environ.get("LLIGMA_TUTOR_TEST_ABORT_BARRIER")
    if not raw:
        return None
    try:
        count = int(raw)
    except ValueError:
        return None
    if count < 2:
        return None
    return threading.Barrier(count)


_ABORT_BARRIER = _barrier_from_env()
_ABORT_LINE_DELAY = _seconds_from_env("LLIGMA_TUTOR_TEST_ABORT_YIELD_MS", 50.0)
_SSE_DELAY = _seconds_from_env("LLIGMA_TUTOR_TEST_SSE_DELAY_MS", 1000.0)


def _abort_trace(frame, event, arg):
    """Dorme subito prima dello ``STORE_ATTR`` del contatore.

    Il delay a livello di riga non basta: l'assegnazione ``x = getattr(...) + 1``
    e' una sola riga, quindi il trace non si ferma tra lettura e scrittura.
    L'evento opcode permette di fermarsi dopo il calcolo e prima dello store;
    tutti gli handler leggono cosi' lo stesso valore quando il lock manca.
    """
    if frame.f_code.co_name != "_note_stream_abort":
        return None
    if event == "call":
        frame.f_trace_opcodes = True
        return _abort_trace
    if event == "opcode" and _ABORT_LINE_DELAY:
        opcode = frame.f_code.co_code[frame.f_lasti]
        if opcode == dis.opmap["STORE_ATTR"]:
            time.sleep(_ABORT_LINE_DELAY)
    return _abort_trace


_original_note = TutorRequestHandler._note_stream_abort


def _instrumented_note_stream_abort(self) -> None:
    if _ABORT_BARRIER is not None:
        try:
            _ABORT_BARRIER.wait(timeout=10)
        except threading.BrokenBarrierError:
            pass
    return _original_note(self)


TutorRequestHandler._note_stream_abort = _instrumented_note_stream_abort

# Il test di disconnessione anticipata chiude intenzionalmente la socket con
# RST dopo il primo evento. BaseHTTPRequestHandler prova poi a leggere la
# richiesta successiva e socketserver registra un traceback per il reset. Il
# reset e' atteso in questo test-only entrypoint: lo consumiamo qui, mentre
# qualsiasi altra eccezione (errori applicativi compresi) continua a
# propagarsi e a far fallire il controllo del log e/o la richiesta.
_original_handle_one_request = TutorRequestHandler.handle_one_request


def _handle_one_request_with_expected_reset(self) -> None:
    try:
        _original_handle_one_request(self)
    except ConnectionResetError:
        self.close_connection = True


TutorRequestHandler.handle_one_request = _handle_one_request_with_expected_reset

if _ABORT_LINE_DELAY:
    # Deve essere impostato prima che ThreadingHTTPServer crei gli handler.
    threading.settrace(_abort_trace)

if _SSE_DELAY:
    _original_write_sse_event = server_module.write_sse_event

    def _delayed_write_sse_event(wfile, event: dict) -> bool:
        written = _original_write_sse_event(wfile, event)
        if written and event.get("type") != "done":
            time.sleep(_SSE_DELAY)
        return written

    server_module.write_sse_event = _delayed_write_sse_event

_switch_interval = os.environ.get("LLIGMA_TUTOR_TEST_SWITCH_INTERVAL")
if _switch_interval:
    try:
        interval = float(_switch_interval)
    except ValueError:
        interval = 0.0
    if interval > 0:
        sys.setswitchinterval(interval)

raise SystemExit(server_module.main())
