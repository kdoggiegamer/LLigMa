"""Eseguibile del comportamento del client web, senza dipendenze Python.

Lo script inline di ``web/tutor/index.html`` viene estratto ed eseguito da
``tests/web_client_harness.js`` contro un DOM e un fetch minimi. Il test copre
la cronologia aggiornata solo con ``done``, lo stream interrotto e l'evento
``error``. Node e' l'unico requisito, gia' usato per la validazione sintattica
del progetto; non viene installato nulla.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "tests" / "web_client_harness.js"


def test_web_client_history_and_error_paths() -> None:
    node = shutil.which("node")
    assert node, "Node non disponibile: impossibile eseguire il test del client web"
    completed = subprocess.run(
        [node, str(HARNESS)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    assert completed.returncode == 0, f"harness client fallito:\n{output}"
    assert "web client harness: PASS" in output, output


def _run_all() -> int:
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