"""Lettura di ``.env`` senza dipendenze esterne.

Il progetto documenta un file ``.env`` per la chiave del docente, ma niente lo
leggeva: avviare il server con la chiave scritta nel file non funzionava, e
l'errore che si vedeva era "backend API non configurato", che manda a cercare il
problema nel posto sbagliato. Questo modulo chiude quel buco.

Regola deliberata: **una variabile gia' presente nell'ambiente vince sempre**.
Così un servizio avviato da systemd o da Docker con la chiave iniettata non viene
sovrascritto da un ``.env`` vecchio rimasto nella cartella.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: str | Path | None = None) -> int:
    """Carica le variabili da ``.env`` e restituisce quante ne ha aggiunte."""
    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if not env_path.is_file():
        return 0

    added = 0
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        # Un valore vuoto nel .env non deve cancellare una variabile d'ambiente.
        if not key or not value or key in os.environ:
            continue
        os.environ[key] = value
        added += 1
    return added
