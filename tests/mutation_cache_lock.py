"""Mutazione del lock della cache del Tutor (secondo invariante).

Il modulo vive sotto ``tests/`` e viene usato dal comando unico
``tests/mutation_stream_aborts.py``. La mutazione sostituisce i tre
``with self._lock:`` di ``AnswerCache`` con un blocco sempre vero, lasciando
la stessa sequenza di operazioni ma senza mutua esclusione. La suite del
servizio viene eseguita in un vero subprocess pytest; il chiamante crea il
backup e questo modulo ripristina sempre il file nel proprio ``finally``.

Campagna reale di tre esecuzioni consecutive: 15/15 run protette con 23/23 test
passati; 9/9 run mutate fallite con 1 failed/22 passed su
``test_cache_serializes_concurrent_mutations``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVICE = ROOT / "lligma" / "tutor" / "service.py"
SERVICE_TEST = ROOT / "tests" / "test_tutor_service.py"
LOCK_COUNT = 3
EXPECTED_PASSED = 23
SERIALIZATION_TEST = "test_cache_serializes_concurrent_mutations"
PROTECTED_RUNS = 5
MUTATION_RUNS = 3
MIN_MUTATION_FAILURES = 2
RUN_TIMEOUT_SECONDS = 120

PASSED_RE = re.compile(r"(\d+) passed")
FAILED_RE = re.compile(r"(\d+) failed")


@dataclass
class CacheRunResult:
    exit_code: int | None
    passed: int | None
    failed: int | None
    output: str


def _newline_for(text: str) -> str:
    return "\r\n" if "\r\n" in text else "\n"


def lock_blocks(text: str) -> tuple[str, str]:
    newline = _newline_for(text)
    lock = f"        with self._lock:{newline}"
    mutated = f"        if True:{newline}"
    return lock, mutated


def cache_source_valid(text: str) -> bool:
    lock, mutated = lock_blocks(text)
    return text.count(lock) == LOCK_COUNT and mutated not in text


def apply_cache_mutation(
    service: Path,
    original_text: str,
    lock_block: str,
    mutated_block: str,
) -> None:
    if original_text.count(lock_block) != LOCK_COUNT:
        raise RuntimeError(
            f"mutazione cache non applicata: trovate {original_text.count(lock_block)} "
            f"occorrenze del lock invece di {LOCK_COUNT}"
        )
    mutated_text = original_text.replace(lock_block, mutated_block)
    if mutated_text.count(lock_block) != 0 or mutated_text.count(mutated_block) != LOCK_COUNT:
        raise RuntimeError("mutazione cache incompleta")
    service.write_bytes(mutated_text.encode("utf-8"))


def restore_cache_and_verify(
    service: Path,
    backup: Path,
    original_bytes: bytes,
    lock_block: str,
    mutated_block: str,
    label: str,
) -> bool:
    shutil.copy2(backup, service)
    final_bytes = service.read_bytes()
    final_text = final_bytes.decode("utf-8")
    byte_equal = final_bytes == original_bytes
    lock_count = final_text.count(lock_block)
    mutation_present = mutated_block in final_text

    print(f"CACHE RESTORE CHECK [{label}]:")
    print(f"  byte_equal_backup={byte_equal}")
    print(f"  lock_occurrences={lock_count}")
    print(f"  mutation_present={mutation_present}")
    print("  cache excerpt:")
    for line in final_text.splitlines():
        if "with self._lock:" in line or "if True:" in line:
            print(f"    {line}")
    return byte_equal and lock_count == LOCK_COUNT and not mutation_present


def run_cache_suite(pytest_python: str) -> CacheRunResult:
    try:
        completed = subprocess.run(
            [
                pytest_python,
                "-m",
                "pytest",
                "-q",
                "--tb=short",
                "-p",
                "no:cacheprovider",
                str(SERVICE_TEST),
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=RUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        output = error.stdout or error.stderr or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        return CacheRunResult(124, None, None, output)

    output = (completed.stdout or "") + (completed.stderr or "")
    passed_match = PASSED_RE.search(output)
    failed_match = FAILED_RE.search(output)
    return CacheRunResult(
        completed.returncode,
        int(passed_match.group(1)) if passed_match else None,
        int(failed_match.group(1)) if failed_match else None,
        output,
    )


def print_cache_result(label: str, index: int, result: CacheRunResult) -> None:
    print(
        f"cache {label} run {index}: exit={result.exit_code} "
        f"passed={result.passed} failed={result.failed}"
    )
    for line in result.output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            "passed" in stripped
            or "failed" in stripped
            or "FAILED" in stripped
            or SERIALIZATION_TEST in stripped
            or "AssertionError" in stripped
        ):
            print(f"  {stripped}")
