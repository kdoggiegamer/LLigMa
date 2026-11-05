"""Caso pytest usato dalle fault-injection del comando di mutazione.

Non fa parte della suite ordinaria: viene invocato esplicitamente, in un
subprocess, da ``mutation_fault_harness.run_fault_scenario`` mentre il server
di produzione contiene la mutazione. Il test reale solleva l'eccezione scelta
dall'ambiente; pytest termina quindi con un exit code non-zero.
"""

from __future__ import annotations

import os

import pytest


def test_fault_scenario() -> None:
    mode = os.environ.get("LLIGMA_MUTATION_FAULT_MODE")
    if mode == "exception":
        raise RuntimeError("fault injection: eccezione esplicita nella run mutata")
    if mode == "interrupt":
        raise KeyboardInterrupt("fault injection: interruzione simulata")
    pytest.fail(f"modalita' di fault non riconosciuta: {mode!r}")
