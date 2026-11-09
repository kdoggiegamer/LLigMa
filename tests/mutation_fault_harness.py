"""Fault-injection reali per il comando di mutazione di ``stream_aborts``.

Il modulo vive interamente sotto ``tests/``. Per ogni scenario:

1. applica la mutazione a ``lligma/tutor/server.py``;
2. invoca ``python -m pytest`` in un vero subprocess sul caso
   ``tests/mutation_fault_case.py``, che solleva l'eccezione richiesta;
3. cattura exit code e output del subprocess;
4. ripristina il file dal backup nel proprio ``finally`` e verifica
   byte-identita, lock presente e mutazione assente.

Il server di produzione non viene modificato e nessuna eccezione viene
simulata in-process dal comando.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAULT_TEST = ROOT / "tests" / "mutation_fault_case.py"
FAULT_TIMEOUT_SECONDS = 120
EXPECTED_FAULT_EXCEPTIONS = {
    "exception": "RuntimeError",
    "interrupt": "KeyboardInterrupt",
}


@dataclass
class FaultResult:
    label: str
    mode: str
    exit_code: int | None
    caught: str | None
    output: str
    restored: bool

    @property
    def expected_exception(self) -> str:
        return EXPECTED_FAULT_EXCEPTIONS[self.mode]

    @property
    def ok(self) -> bool:
        return (
            self.exit_code is not None
            and self.exit_code != 0
            and self.caught == self.expected_exception
            and self.restored
        )


def _newline_for(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def lock_blocks(text: str) -> tuple[str, str]:
    newline = _newline_for(text)
    lock = (
        f"        with server.stream_aborts_lock:{newline}"
        f"            server.stream_aborts = getattr(server, \"stream_aborts\", 0) + 1{newline}"
    )
    mutated = (
        f"        server.stream_aborts = getattr(server, \"stream_aborts\", 0) + 1{newline}"
    )
    return lock, mutated


def mutation_present(text: str, mutated_block: str) -> bool:
    """Riconosce la riga mutata solo all'indentazione di 8 spazi."""
    return text.startswith(mutated_block) or f"\n{mutated_block}" in text


def apply_mutation(
    server: Path,
    original_text: str,
    lock_block: str,
    mutated_block: str,
) -> None:
    mutated_text = original_text.replace(lock_block, mutated_block, 1)
    if mutated_text == original_text:
        raise RuntimeError("mutazione non applicata: blocco lock non trovato")
    server.write_bytes(mutated_text.encode("utf-8"))


def restore_and_verify(
    server: Path,
    backup: Path,
    original_bytes: bytes,
    lock_block: str,
    mutated_block: str,
    label: str,
) -> bool:
    shutil.copy2(backup, server)
    final_bytes = server.read_bytes()
    final_text = final_bytes.decode("utf-8")
    lock_present = lock_block in final_text
    mutation_present = f"\n{mutated_block}" in f"\n{final_text}"
    byte_equal = final_bytes == original_bytes

    print(f"RESTORE CHECK [{label}]:")
    print(f"  byte_equal_backup={byte_equal}")
    print(f"  lock_present={lock_present}")
    print(f"  mutation_present={mutation_present}")
    marker = "    def _note_stream_abort(self) -> None:"
    start = final_text.find(marker)
    if start >= 0:
        print("  counter excerpt:")
        for line in final_text[start:].splitlines()[:6]:
            print(f"    {line}")
    return byte_equal and lock_present and not mutation_present


def resolve_pytest_python() -> str:
    """Trova un interprete con pytest gia' disponibile, senza installarlo."""
    candidates = [
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            check = subprocess.run(
                [str(candidate), "-m", "pytest", "--version"],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if check.returncode == 0:
            return str(candidate)
    raise RuntimeError(
        "pytest non disponibile: installare pytest nell'ambiente di test "
        "(la .venv del progetto) prima di eseguire le fault-injection."
    )


def _invoke_pytest(pytest_python: str, mode: str) -> tuple[int | None, str]:
    env = os.environ.copy()
    env["LLIGMA_MUTATION_FAULT_MODE"] = mode
    command = [
        pytest_python,
        "-m",
        "pytest",
        "-q",
        "--tb=short",
        "-p",
        "no:cacheprovider",
        str(FAULT_TEST),
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=FAULT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or error.stderr or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return 124, output
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output


def run_fault_scenario(
    *,
    label: str,
    mode: str,
    server: Path,
    backup: Path,
    original_bytes: bytes,
    original_text: str,
    lock_block: str,
    mutated_block: str,
    pytest_python: str,
) -> FaultResult:
    """Esegue un fault reale e ripristina il server nel ``finally``."""
    result = FaultResult(
        label=label,
        mode=mode,
        exit_code=None,
        caught=None,
        output="",
        restored=False,
    )
    try:
        apply_mutation(server, original_text, lock_block, mutated_block)
        print(
            f"FAULT {label}: mutazione applicata; "
            f"invocazione pytest reale in subprocess (mode={mode})"
        )
        exit_code, output = _invoke_pytest(pytest_python, mode)
        result.exit_code = exit_code
        result.output = output
        for exception_name in ("RuntimeError", "KeyboardInterrupt"):
            if exception_name in output:
                result.caught = exception_name
                break
        print(
            f"FAULT {label}: pytest exit={result.exit_code} "
            f"caught={result.caught} expected={result.expected_exception}"
        )
        for line in output.splitlines():
            if line.strip():
                print(f"  {line}")
    finally:
        result.restored = restore_and_verify(
            server,
            backup,
            original_bytes,
            lock_block,
            mutated_block,
            label,
        )
    return result
