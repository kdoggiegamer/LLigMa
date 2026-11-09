#!/usr/bin/env python3
"""Verifica automatica degli scenari critici del comando di mutazione.

Uso, dalla radice del progetto::

    python tests/mutation_safety.py

Non fa parte della suite pytest. Il comando e' non interattivo e, in caso di
errore, tenta sempre di ripristinare i byte iniziali, rimuovere transazioni e
lock e terminare i processi figli che ha avviato.

Scenari esercitati:

1. meccanismo out-of-tree assente (variabile test-only
   ``LLIGMA_MUTATION_DISABLE_OUT_OF_TREE=1``): il comando completo deve fallire
   prima di applicare la mutazione, senza lasciare transazioni;
2. transazione stale con marker clean valido: il watchdog ancora vivo deve
   rimuoverla e terminare;
3. PID del padre riutilizzato: il recovery non deve bloccare una transazione
   orfana solo perche' il PID e' ora di un altro processo;
4. recovery concorrente: un secondo recovery non deve toccare una transazione
   mentre il primo detiene il lock; dopo il rilascio il recovery deve riuscire;
5. albero del comando ucciso con mutazione applicata: il watchdog WMI deve
   sopravvivere, ripristinare entrambi i file e terminare;
6. transazione orfana con backup: il comando successivo deve fare RECOVERY e
   arrivare a VERDICT: PASS.

Ogni scenario verifica gli hash originali e l'assenza di transazioni, backup,
lock, watchdog o log di test residui. Il comando stampa un solo esito finale
``MUTATION_SAFETY: PASS`` oppure ``MUTATION_SAFETY: FAIL``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "lligma" / "tutor" / "server.py"
SERVICE = ROOT / "lligma" / "tutor" / "service.py"
MUTATION = ROOT / "tests" / "mutation_stream_aborts.py"

sys.path.insert(0, str(ROOT / "tests"))
import mutation_interruption as demo  # noqa: E402
from mutation_transaction import (  # noqa: E402
    RECOVERY_LOCK_PREFIX,
    RecoveryLock,
    atomic_write,
    process_alive,
    process_start_time,
    recover_pending,
)

CHILD_CODE = r"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, 'tests')
from mutation_transaction import MutationTransaction, write_json
from mutation_fault_harness import apply_mutation, lock_blocks
from mutation_cache_lock import apply_cache_mutation, lock_blocks as cache_lock_blocks

server = Path('lligma/tutor/server.py')
service = Path('lligma/tutor/service.py')
transaction = MutationTransaction.begin(server, service)

if os.environ.get('SAFETY_MUTATE_SERVER') == '1':
    text = server.read_text(encoding='utf-8')
    lock, mutated = lock_blocks(text)
    apply_mutation(server, text, lock, mutated)

if os.environ.get('SAFETY_MUTATE_SERVICE') == '1':
    text = service.read_text(encoding='utf-8')
    lock, mutated = cache_lock_blocks(text)
    apply_cache_mutation(service, text, lock, mutated)

if os.environ.get('SAFETY_CLEAN') == '1':
    write_json(
        transaction.path / 'clean.json',
        {'status': 'clean', 'files': transaction.manifest['files']},
    )

Path(os.environ['SAFETY_MARKER']).write_text(
    json.dumps(
        {
            'transaction': str(transaction.path),
            'watchdog': transaction.watchdog.pid,
            'parent_start': transaction.manifest.get('parent_start'),
        }
    ),
    encoding='utf-8',
)
print('safety-child-ready', flush=True)
time.sleep(120)
"""

WATCHDOG_PIDS: list[int] = []


def hashes() -> tuple[str, str]:
    return (
        hashlib.sha256(SERVER.read_bytes()).hexdigest(),
        hashlib.sha256(SERVICE.read_bytes()).hexdigest(),
    )


def assert_hashes(expected: tuple[str, str], label: str) -> None:
    actual = hashes()
    if actual != expected:
        raise AssertionError(f"{label}: hash diversi: {actual} != {expected}")


def project_transactions() -> list[Path]:
    return demo.project_transactions()


def recovery_locks() -> list[Path]:
    return sorted(ROOT.glob(RECOVERY_LOCK_PREFIX + "*.lock"))


def assert_clean(expected: tuple[str, str], label: str) -> None:
    assert_hashes(expected, label)
    transactions = project_transactions()
    if transactions:
        raise AssertionError(f"{label}: transazioni residue: {transactions}")
    locks = recovery_locks()
    if locks:
        raise AssertionError(f"{label}: lock di recovery residui: {locks}")
    alive = [pid for pid in WATCHDOG_PIDS if process_alive(pid)]
    if alive:
        raise AssertionError(f"{label}: watchdog ancora vivi: {alive}")


def kill_pid(pid: int) -> None:
    demo.kill_pid(pid)


def wait_until(predicate, timeout: float, description: str) -> None:
    demo.wait_until(predicate, timeout, description)


def spawn_stale(
    *,
    mutate_server: bool,
    mutate_service: bool,
    clean: bool,
    leave_watchdog: bool,
) -> tuple[Path, int]:
    """Crea una transazione stale e restituisce percorso e PID watchdog."""
    with tempfile.TemporaryDirectory(prefix="lligma-safety-child-") as directory:
        marker = Path(directory) / "ready.json"
        log_path = Path(directory) / "child.log"
        env = os.environ.copy()
        env.update(
            {
                "SAFETY_MARKER": str(marker),
                "SAFETY_MUTATE_SERVER": "1" if mutate_server else "0",
                "SAFETY_MUTATE_SERVICE": "1" if mutate_service else "0",
                "SAFETY_CLEAN": "1" if clean else "0",
            }
        )
        with log_path.open("w+", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                [sys.executable, "-c", CHILD_CODE],
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            try:
                wait_until(marker.exists, 30, "transazione stale pronta")
                payload = json.loads(marker.read_text(encoding="utf-8"))
                transaction = Path(payload["transaction"])
                watchdog_pid = int(payload["watchdog"])
                if watchdog_pid:
                    WATCHDOG_PIDS.append(watchdog_pid)
                if not transaction.is_dir():
                    raise AssertionError(f"transazione non creata: {transaction}")
                if not process_alive(watchdog_pid):
                    raise AssertionError(f"watchdog non attivo: {watchdog_pid}")

                if leave_watchdog:
                    # Il padre muore con un clean marker gia' scritto: il
                    # watchdog deve rimuovere la transazione e uscire da solo.
                    kill_pid(process.pid)
                    process.wait(timeout=20)
                    wait_until(
                        lambda: not transaction.exists()
                        and not process_alive(watchdog_pid),
                        30,
                        "watchdog con clean marker",
                    )
                else:
                    kill_pid(watchdog_pid)
                    wait_until(
                        lambda: not process_alive(watchdog_pid),
                        10,
                        "watchdog stale terminato",
                    )
                    kill_pid(process.pid)
                    process.wait(timeout=20)
                    if not transaction.is_dir():
                        raise AssertionError("transazione sparita troppo presto")
                return transaction, watchdog_pid
            finally:
                if process.poll() is None:
                    kill_pid(process.pid)
                    process.wait(timeout=20)
                log.flush()
                log.seek(0)
                output = log.read()
                if "Traceback" in output or "Error" in output:
                    print(output)


def run_recovery_child() -> tuple[int, str]:
    code = (
        "import sys; sys.path.insert(0, 'tests'); "
        "from mutation_transaction import recover_pending; "
        "r, e = recover_pending(); print('RECOVERED', r); print('ERRORS', e); "
        "raise SystemExit(1 if e else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output


def scenario_out_of_tree_unavailable(before: tuple[str, str]) -> None:
    env = os.environ.copy()
    env["LLIGMA_MUTATION_DISABLE_OUT_OF_TREE"] = "1"
    # Popen e non subprocess.run: se la guardia viene rotta (e' il caso del
    # controllo negativo) il comando non esce con 2 e puo' durare quanto la
    # campagna intera. Al timeout l'albero va terminato davvero, altrimenti
    # restano pytest figli che scrivono nel progetto, e il FAIL deve riportare
    # lo stesso messaggio del caso positivo invece di un traceback di
    # orchestrazione: il timeout e' la prova che la guardia non e' scattata.
    timeout = 240
    process = subprocess.Popen(
        [sys.executable, str(MUTATION)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=(os.name != "nt"),
    )
    timed_out = False
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        demo.kill_tree(process.pid)
        try:
            output, _ = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            output, _ = process.communicate()
    output = output or ""
    print(output[-4000:])
    if timed_out:
        raise AssertionError(
            "out-of-tree assente: atteso exit 2, ottenuto un comando ancora "
            f"in esecuzione dopo {timeout}s (guardia inerte)"
        )
    if process.returncode != 2:
        raise AssertionError(
            f"out-of-tree assente: atteso exit 2, ottenuto {process.returncode}"
        )
    if "out-of-tree" not in output and "impossibile creare la transazione" not in output:
        raise AssertionError("out-of-tree assente: errore atteso non trovato")
    assert_hashes(before, "out-of-tree assente")
    if project_transactions():
        raise AssertionError("out-of-tree assente: transazione residua")
    print("OUT_OF_TREE_UNAVAILABLE: PASS")


def scenario_clean_stale(before: tuple[str, str]) -> None:
    transaction, watchdog_pid = spawn_stale(
        mutate_server=False,
        mutate_service=False,
        clean=True,
        leave_watchdog=True,
    )
    if transaction.exists():
        raise AssertionError("clean stale: transazione non rimossa")
    if process_alive(watchdog_pid):
        raise AssertionError("clean stale: watchdog ancora vivo")
    assert_clean(before, "clean stale")
    print("STALE_CLEAN_RECOVERY: PASS")


def scenario_pid_reuse(before: tuple[str, str]) -> None:
    transaction, watchdog_pid = spawn_stale(
        mutate_server=True,
        mutate_service=False,
        clean=False,
        leave_watchdog=False,
    )
    try:
        demo.assert_mutation_active()
        manifest_path = transaction / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["parent_pid"] = os.getpid()
        parent_start = process_start_time(os.getpid())
        if parent_start is None:
            raise AssertionError("PID reuse: impossibile leggere l'identita' del test")
        manifest["parent_start"] = parent_start + 1
        atomic_write(
            manifest_path,
            json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8"),
        )
        recovered, errors = recover_pending()
        print("PID_REUSE_RECOVERY_OUTPUT:", recovered, errors)
        if errors:
            raise AssertionError(f"PID reuse: recovery ha fallito: {errors}")
        assert_clean(before, "PID reuse")
        if transaction.exists():
            raise AssertionError("PID reuse: transazione residua")
    finally:
        if process_alive(watchdog_pid):
            kill_pid(watchdog_pid)
    print("PID_REUSE_RECOVERY: PASS")


def scenario_concurrent_recovery(before: tuple[str, str]) -> None:
    transaction, watchdog_pid = spawn_stale(
        mutate_server=True,
        mutate_service=True,
        clean=False,
        leave_watchdog=False,
    )
    lock = RecoveryLock(ROOT / f"{RECOVERY_LOCK_PREFIX}{transaction.name}.lock")
    try:
        demo.assert_mutation_active()
        service_text = SERVICE.read_text(encoding="utf-8")
        if "with self._lock:" in service_text:
            raise AssertionError("concorrenza: service.py non e' mutato")
        if not lock.acquire():
            raise AssertionError("concorrenza: impossibile acquisire il lock di test")
        exit_code, output = run_recovery_child()
        print(output[-3000:])
        if exit_code == 0:
            raise AssertionError("concorrenza: il secondo recovery ha avuto successo")
        if "recovery concorrente" not in output:
            raise AssertionError("concorrenza: errore di lock non rilevato")
        if not transaction.is_dir():
            raise AssertionError("concorrenza: transazione rimossa dal secondo recovery")
        if hashes() == before:
            raise AssertionError("concorrenza: file gia' ripristinati prima del lock")
    finally:
        lock.release()
        if process_alive(watchdog_pid):
            kill_pid(watchdog_pid)

    exit_code, output = run_recovery_child()
    print(output[-3000:])
    if exit_code != 0:
        raise AssertionError(f"concorrenza: recovery finale exit {exit_code}")
    assert_clean(before, "concorrenza")
    if transaction.exists():
        raise AssertionError("concorrenza: transazione finale residua")
    print("CONCURRENT_RECOVERY: PASS")


def run(before: tuple[str, str]) -> None:
    try:
        demo.interrupt_full_command(before)
        assert_clean(before, "tree kill")
    except Exception as error:
        print(f"TREE_KILL: FAIL: {type(error).__name__}: {error}", flush=True)
        if not isinstance(error, AssertionError):
            traceback.print_exc()
        raise
    print("TREE_KILL: PASS")

    try:
        orphan = demo.make_orphan_transaction()
        demo.run_full_after_orphan(before, orphan)
        assert_clean(before, "orphan full")
    except Exception as error:
        print(f"ORPHAN_RECOVERY: FAIL: {type(error).__name__}: {error}", flush=True)
        if not isinstance(error, AssertionError):
            traceback.print_exc()
        raise
    print("ORPHAN_RECOVERY: PASS")


def cleanup(before: tuple[str, str], original: dict[Path, bytes]) -> bool:
    ok = True
    for pid in list(WATCHDOG_PIDS):
        if process_alive(pid):
            kill_pid(pid)
    recovered, errors = recover_pending()
    for message in recovered:
        print(f"CLEANUP_RECOVERY: {message}")
    if errors:
        ok = False
        for message in errors:
            print(f"CLEANUP_ERROR: {message}")
    if hashes() != before:
        for path, data in original.items():
            atomic_write(path, data)
        print("CLEANUP: byte iniziali ripristinati dal test harness")
    # Dopo un errore non deve restare alcuna transazione o lock, anche se il
    # recovery ha segnalato un backup mancante.
    for transaction in project_transactions():
        shutil.rmtree(transaction, ignore_errors=True)
        print(f"CLEANUP: rimossa transazione residua {transaction.name}")
    for lock_path in recovery_locks():
        try:
            lock_path.unlink()
        except OSError:
            pass
    try:
        assert_clean(before, "cleanup")
    except AssertionError as error:
        print(f"CLEANUP_ERROR: {error}")
        ok = False
    return ok


def main() -> int:
    before = hashes()
    original = {SERVER: SERVER.read_bytes(), SERVICE: SERVICE.read_bytes()}
    recovered, errors = recover_pending()
    for message in recovered:
        print(f"PRE_RECOVERY: {message}")
    if errors:
        for message in errors:
            print(f"PRE_RECOVERY_ERROR: {message}")
        print("MUTATION_SAFETY: FAIL")
        return 1
    if hashes() != before:
        print("MUTATION_SAFETY: FAIL: hash iniziali cambiati dal recovery")
        return 1

    print(f"SNAPSHOT: server={before[0]} service={before[1]}")
    status = 0
    scenarios = [
        ("OUT_OF_TREE_UNAVAILABLE", scenario_out_of_tree_unavailable),
        ("STALE_CLEAN_RECOVERY", scenario_clean_stale),
        ("PID_REUSE_RECOVERY", scenario_pid_reuse),
        ("CONCURRENT_RECOVERY", scenario_concurrent_recovery),
    ]
    try:
        for label, scenario in scenarios:
            try:
                scenario(before)
            except Exception as error:
                print(f"{label}: FAIL: {type(error).__name__}: {error}", flush=True)
                if not isinstance(error, AssertionError):
                    traceback.print_exc()
                status = 1
                break
        if status == 0:
            run(before)
    except Exception as error:
        if not isinstance(error, AssertionError):
            traceback.print_exc()
        status = 1
    finally:
        if not cleanup(before, original):
            status = 1
    print(f"MUTATION_SAFETY: {'PASS' if status == 0 else 'FAIL'}")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
