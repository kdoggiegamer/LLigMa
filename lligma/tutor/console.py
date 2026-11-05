"""Output UTF-8 affidabile sui terminali Windows (cp1252 di default)."""

from __future__ import annotations

import sys


def setup_utf8() -> None:
    """Riconfigura stdout/stderr senza mai far fallire l'avvio."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass
