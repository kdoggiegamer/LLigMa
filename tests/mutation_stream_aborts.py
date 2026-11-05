"""Esperimento ripetibile di mutazione per il lock di ``stream_aborts``.

Uso, dalla radice del progetto::

    python tests/mutation_stream_aborts.py

Il comando verifica due invarianti. Per ``stream_aborts`` esegue il test
concorrente E2E cinque volte con il lock reale, poi sostituisce temporaneamente
il solo incremento protetto con un incremento non atomico ed esegue tre run
mutate. Per ``AnswerCache._lock`` esegue cinque volte la suite del servizio con
pytest reale, poi sostituisce i tre ``with self._lock:`` con un blocco sempre
vero ed esegue tre run mutate. Ogni fase ha un ``finally`` che ricopia il backup
e verifica byte-identita, lock presente e mutazione assente. Infine esegue due
fault-injection reali: ogni fault applica la mutazione e lancia un vero
subprocess ``python -m pytest`` sul caso ``tests/mutation_fault_case.py``; uno
solleva ``RuntimeError`` nel test, l'altro ``KeyboardInterrupt`` durante la run.

Esce con 0 solo se tutte le cinque run protette riportano il delta esatto, la
mutazione fallisce in almeno due delle tre run con ``stream_aborts`` osservato,
e entrambi i fault pytest terminano con exit code non-zero, mostrano
l'eccezione attesa e ripristinano correttamente il file.

Campagna reale di tre esecuzioni consecutive: stream 15/15 run protette con
delta 100 e 9/9 mutate fallite con valori 3, 4, 4; 2, 3, 2; 3, 3, 4; cache
15/15 run con 23/23 test e 9/9 mutate fallite con 1 failed/22 passed; fault
``RuntimeError`` exit 1 e ``KeyboardInterrupt`` exit 2 in tutte e tre, con
ripristino verificato. Nella verifica WMI le mutate stream hanno osservato
96, 86 e 89, sempre sotto 100, e il comando ha chiuso con ``VERDICT: PASS``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from mutation_cache_lock import (
    EXPECTED_PASSED as CACHE_EXPECTED_PASSED,
    MIN_MUTATION_FAILURES as CACHE_MIN_MUTATION_FAILURES,
    MUTATION_RUNS as CACHE_MUTATION_RUNS,
    PROTECTED_RUNS as CACHE_PROTECTED_RUNS,
    SERIALIZATION_TEST as CACHE_SERIALIZATION_TEST,
    apply_cache_mutation,
    cache_source_valid,
    lock_blocks as cache_lock_blocks,
    print_cache_result,
    restore_cache_and_verify,
    run_cache_suite,
)
from mutation_fault_harness import (
    apply_mutation,
    lock_blocks,
    mutation_present,
    resolve_pytest_python,
    restore_and_verify,
    run_fault_scenario,
)
from mutation_transaction import MutationTransaction, atomic_write, recover_pending

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "lligma" / "tutor" / "server.py"
CACHE_SERVICE = ROOT / "lligma" / "tutor" / "service.py"
E2E = ROOT / "tests" / "test_tutor_server_e2e.py"

EXPECTED_ABORTS = 100
PROTECTED_RUNS = 5
MUTATION_RUNS = 3
MIN_MUTATION_FAILURES = 2
RUN_TIMEOUT_SECONDS = 120

DELTA_RE = re.compile(r"stream_aborts delta=(\d+) \(atteso (\d+)\)")
OBSERVED_RE = re.compile(r"stream_aborts=(\d+), atteso almeno")
ANY_ABORTS_RE = re.compile(r"stream_aborts=(\d+)")


@dataclass
class RunResult:
    exit_code: int
    delta: int | None
    observed: int | None
    output: str


def _parse(output: str) -> tuple[int | None, int | None]:
    delta_match = DELTA_RE.search(output)
    delta = int(delta_match.group(1)) if delta_match else None

    observed_match = OBSERVED_RE.search(output)
    if observed_match:
        observed = int(observed_match.group(1))
    else:
        any_match = ANY_ABORTS_RE.search(output)
        observed = int(any_match.group(1)) if any_match else None
    return delta, observed


def _run_e2e() -> RunResult:
    try:
        completed = subprocess.run(
            [sys.executable, str(E2E)],
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
        return RunResult(124, None, None, output)

    output = completed.stdout or ""
    delta, observed = _parse(output)
    return RunResult(completed.returncode, delta, observed, output)


def _pause_for_interruption(stage: str) -> None:
    """Punto di pausa deterministico per il test di interruzione."""
    if os.environ.get("LLIGMA_MUTATION_PAUSE_AFTER") != stage:
        return
    marker = os.environ.get("LLIGMA_MUTATION_PAUSE_FILE")
    if marker:
        Path(marker).write_text(f"{stage}\n", encoding="utf-8")
    while True:
        time.sleep(0.05)


def _restore_known_mutation() -> bool:
    """Ripristina una mutazione nota, ma senza fingere un hash verificato."""
    server_text = SERVER.read_text(encoding="utf-8")
    service_text = CACHE_SERVICE.read_text(encoding="utf-8")
    server_lock, server_mutated = lock_blocks(server_text)
    cache_lock, cache_mutated = cache_lock_blocks(service_text)
    server_dirty = mutation_present(server_text, server_mutated)
    cache_dirty = cache_mutated in service_text
    if not server_dirty and not cache_dirty:
        return True

    if server_dirty:
        if server_text.count(server_lock) != 0 or server_text.count(server_mutated) != 1:
            print("ERRORE: server.py mutato in una forma non riconosciuta; nessun ripristino.")
            return False
        repaired = server_text.replace(server_mutated, server_lock, 1)
        if server_lock not in repaired or mutation_present(repaired, server_mutated):
            print("ERRORE: ripristino strutturale di server.py non riuscito.")
            return False
        atomic_write(SERVER, repaired.encode("utf-8"))
        print("RECOVERY: server.py conteneva la mutazione nota; lock ripristinato.")

    if cache_dirty:
        if service_text.count(cache_lock) != 0 or service_text.count(cache_mutated) != 3:
            print("ERRORE: service.py mutato in una forma non riconosciuta; nessun ripristino.")
            return False
        repaired = service_text.replace(cache_mutated, cache_lock)
        if repaired.count(cache_lock) != 3 or cache_mutated in repaired:
            print("ERRORE: ripristino strutturale di service.py non riuscito.")
            return False
        atomic_write(CACHE_SERVICE, repaired.encode("utf-8"))
        print("RECOVERY: service.py conteneva la mutazione nota; tre lock ripristinati.")

    print(
        "ERRORE: ripristino strutturale eseguito senza backup/hash originale; "
        "byte-identita' non verificabile. Verdetto FAIL."
    )
    print("VERDICT: FAIL")
    return False


def _print_result(label: str, index: int, result: RunResult) -> None:
    print(f"{label} run {index}: exit={result.exit_code} delta={result.delta} observed={result.observed}")
    for line in result.output.splitlines():
        stripped = line.strip()
        if stripped and (
            "stream_aborts" in stripped
            or "test superati" in stripped
            or stripped.startswith("FAIL ")
            or "Traceback" in stripped
        ):
            print(f"  {stripped}")


def main() -> int:
    recovered, recovery_errors = recover_pending()
    for message in recovered:
        print(f"RECOVERY: {message}")
    if recovery_errors:
        for message in recovery_errors:
            print(f"ERRORE: {message}")
        print("VERDICT: FAIL")
        return 2
    if not _restore_known_mutation():
        return 2

    original_bytes = SERVER.read_bytes()
    original_text = original_bytes.decode("utf-8")
    lock_block, mutated_block = lock_blocks(original_text)

    service_original_bytes = CACHE_SERVICE.read_bytes()
    service_original_text = service_original_bytes.decode("utf-8")
    cache_lock_block, cache_mutated_block = cache_lock_blocks(service_original_text)

    if lock_block not in original_text or mutation_present(original_text, mutated_block):
        print("ERRORE: il server non e' nella versione protetta attesa; nessuna mutazione applicata.")
        return 2
    if not cache_source_valid(service_original_text):
        print("ERRORE: il lock della cache non e' nella forma attesa; nessuna mutazione applicata.")
        return 2

    try:
        pytest_python = resolve_pytest_python()
    except RuntimeError as error:
        print(f"ERRORE: {error}")
        return 2
    print(f"PYTEST_PYTHON={pytest_python}")

    try:
        transaction = MutationTransaction.begin(SERVER, CACHE_SERVICE)
    except Exception as error:
        print(f"ERRORE: impossibile creare la transazione: {type(error).__name__}: {error}")
        print("VERDICT: FAIL")
        return 2

    backup = transaction.server_backup
    service_backup = transaction.service_backup
    protected_results: list[RunResult] = []
    mutation_results: list[RunResult] = []
    cache_protected_results: list = []
    cache_mutation_results: list = []
    normal_restore_ok = False
    cache_restore_ok = False
    cache_outer_restore_ok = False
    cache_phase_error = False
    fault_results: list[bool] = []
    outer_restore_ok = False
    cleanup_ok = False
    try:
        print("=== PROTECTED RUNS (lock reale) ===")
        for index in range(1, PROTECTED_RUNS + 1):
            result = _run_e2e()
            protected_results.append(result)
            _print_result("protected", index, result)

        print("=== MUTATED RUNS (incremento non atomico) ===")
        try:
            apply_mutation(SERVER, original_text, lock_block, mutated_block)
            _pause_for_interruption("server")
            for index in range(1, MUTATION_RUNS + 1):
                result = _run_e2e()
                mutation_results.append(result)
                _print_result("mutated", index, result)
        finally:
            normal_restore_ok = restore_and_verify(
                SERVER,
                backup,
                original_bytes,
                lock_block,
                mutated_block,
                "normal mutation",
            )

        print("=== SECONDO INVARIANTE: CACHE LOCK (suite pytest reale) ===")
        try:
            print("--- cache protected runs ---")
            for index in range(1, CACHE_PROTECTED_RUNS + 1):
                result = run_cache_suite(pytest_python)
                cache_protected_results.append(result)
                print_cache_result("protected", index, result)

            print("--- cache mutated runs ---")
            try:
                apply_cache_mutation(
                    CACHE_SERVICE,
                    service_original_text,
                    cache_lock_block,
                    cache_mutated_block,
                )
                _pause_for_interruption("cache")
                for index in range(1, CACHE_MUTATION_RUNS + 1):
                    result = run_cache_suite(pytest_python)
                    cache_mutation_results.append(result)
                    print_cache_result("mutated", index, result)
            finally:
                cache_restore_ok = restore_cache_and_verify(
                    CACHE_SERVICE,
                    service_backup,
                    service_original_bytes,
                    cache_lock_block,
                    cache_mutated_block,
                    "cache normal mutation",
                )
        except Exception as error:
            print(f"CACHE MUTATION ERROR: {type(error).__name__}: {error}")
            cache_phase_error = True

        print("=== FAULT INJECTION (veri subprocess pytest) ===")
        fault_specs = [
            ("explicit_exception", "exception"),
            ("simulated_interrupt", "interrupt"),
        ]
        for label, mode in fault_specs:
            try:
                outcome = run_fault_scenario(
                    label=label,
                    mode=mode,
                    server=SERVER,
                    backup=backup,
                    original_bytes=original_bytes,
                    original_text=original_text,
                    lock_block=lock_block,
                    mutated_block=mutated_block,
                    pytest_python=pytest_python,
                )
            except Exception as error:
                print(f"FAULT {label}: ERRORE harness: {type(error).__name__}: {error}")
                fault_results.append(False)
                continue

            print(f"FAULT {label}: restored={outcome.restored}")
            fault_results.append(outcome.ok)
    finally:
        shutil.copy2(backup, SERVER)
        outer_restore_ok = restore_and_verify(
            SERVER,
            backup,
            original_bytes,
            lock_block,
            mutated_block,
            "outer finally",
        )
        shutil.copy2(service_backup, CACHE_SERVICE)
        cache_outer_restore_ok = restore_cache_and_verify(
            CACHE_SERVICE,
            service_backup,
            service_original_bytes,
            cache_lock_block,
            cache_mutated_block,
            "cache outer finally",
        )
        cleanup_ok = transaction.finish(normal_restore_ok and cache_outer_restore_ok)

    protected_exact = sum(
        result.exit_code == 0 and result.delta == EXPECTED_ABORTS
        for result in protected_results
    )
    mutation_failures = sum(
        result.exit_code != 0
        and result.observed is not None
        and result.observed < EXPECTED_ABORTS
        for result in mutation_results
    )
    cache_protected_exact = sum(
        result.exit_code == 0
        and result.passed == CACHE_EXPECTED_PASSED
        and result.failed is None
        for result in cache_protected_results
    )
    cache_mutation_failures = sum(
        result.exit_code != 0
        and result.failed is not None
        and CACHE_SERIALIZATION_TEST in result.output
        for result in cache_mutation_results
    )
    faults_ok = len(fault_results) == 2 and all(fault_results)

    print("=== SUMMARY ===")
    print(
        f"protected: {protected_exact}/{PROTECTED_RUNS} run con delta esatto "
        f"{EXPECTED_ABORTS}"
    )
    print(
        f"mutated: {mutation_failures}/{MUTATION_RUNS} run fallite con "
        f"stream_aborts osservato (richieste almeno {MIN_MUTATION_FAILURES})"
    )
    print(
        f"cache-protected: {cache_protected_exact}/{CACHE_PROTECTED_RUNS} run "
        f"con {CACHE_EXPECTED_PASSED} test passati"
    )
    print(
        f"cache-mutated: {cache_mutation_failures}/{CACHE_MUTATION_RUNS} run "
        f"fallite su {CACHE_SERIALIZATION_TEST} "
        f"(richieste almeno {CACHE_MIN_MUTATION_FAILURES})"
    )
    print(
        "fault-injection: "
        + ", ".join(str(value) for value in fault_results)
        + f" ({sum(fault_results)}/2 ripristinate e fallite realmente)"
    )
    print(f"normal_restore_ok={normal_restore_ok}")
    print(f"cache_restore_ok={cache_restore_ok}")
    print(f"outer_restore_ok={outer_restore_ok}")
    print(f"cache_outer_restore_ok={cache_outer_restore_ok}")
    print(f"transaction_cleanup_ok={cleanup_ok}")

    failures: list[str] = []
    if protected_exact != PROTECTED_RUNS:
        failures.append(
            f"protected {protected_exact}/{PROTECTED_RUNS}, attese {PROTECTED_RUNS} con delta {EXPECTED_ABORTS}"
        )
    if mutation_failures < MIN_MUTATION_FAILURES:
        failures.append(
            f"mutated {mutation_failures}/{MUTATION_RUNS}, richieste almeno {MIN_MUTATION_FAILURES} fallite"
        )
    if cache_protected_exact != CACHE_PROTECTED_RUNS:
        failures.append(
            f"cache-protected {cache_protected_exact}/{CACHE_PROTECTED_RUNS}, "
            f"attesi {CACHE_EXPECTED_PASSED} test passati per run"
        )
    if cache_mutation_failures < CACHE_MIN_MUTATION_FAILURES:
        failures.append(
            f"cache-mutated {cache_mutation_failures}/{CACHE_MUTATION_RUNS}, "
            f"richieste almeno {CACHE_MIN_MUTATION_FAILURES} fallite su {CACHE_SERIALIZATION_TEST}"
        )
    if cache_phase_error:
        failures.append("errore durante la fase di mutazione della cache")
    if not faults_ok:
        failures.append(
            "fault-injection non tutti riusciti: serve exit pytest non-zero, eccezione attesa e ripristino"
        )
    if not normal_restore_ok:
        failures.append("ripristino dopo le run mutate non valido")
    if not cache_restore_ok:
        failures.append("ripristino dopo la mutazione della cache non valido")
    if not outer_restore_ok:
        failures.append("ripristino finale del server non valido")
    if not cache_outer_restore_ok:
        failures.append("ripristino finale del servizio non valido")
    if not cleanup_ok:
        failures.append("transazione non rimossa o ripristino non verificato")

    if failures:
        print("FAILURE_REASONS:")
        for failure in failures:
            print(f"  - {failure}")
    print(f"VERDICT: {'PASS' if not failures else 'FAIL'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
