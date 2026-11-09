#!/usr/bin/env python3
"""Dimostrazione ripetibile dell'interruzione del comando di mutazione.

Non fa parte della suite pytest: viene eseguito esplicitamente dalla radice
del progetto. Lo script:

1. avvia il comando completo e ne uccide l'intero albero dopo che la mutazione
   di ``server.py`` e' stata applicata e il watchdog out-of-tree e' attivo;
2. verifica che il watchdog sopravviva, ripristini gli hash originali, rimuova
   la transazione e termini;
3. crea una seconda transazione orfana uccidendo watchdog e padre, lasciando
   di proposito i file mutati e il backup;
4. esegue di nuovo il comando completo: il recovery all'avvio deve ripristinare
   il backup e il comando deve arrivare a ``VERDICT: PASS``;
5. verifica che la mutazione attiva abbia davvero fatto fallire i test e che
   non restino transazioni, backup o processi.

Uso::

    python tests/mutation_interruption.py
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "lligma" / "tutor" / "server.py"
SERVICE = ROOT / "lligma" / "tutor" / "service.py"
MUTATION = ROOT / "tests" / "mutation_stream_aborts.py"
WATCHDOG = ROOT / "tests" / "mutation_watchdog.py"

sys.path.insert(0, str(ROOT / "tests"))
from mutation_transaction import process_alive, recover_pending  # noqa: E402


def hashes() -> tuple[str, str]:
    return (
        hashlib.sha256(SERVER.read_bytes()).hexdigest(),
        hashlib.sha256(SERVICE.read_bytes()).hexdigest(),
    )


def wait_until(predicate, timeout: float, description: str) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"timeout: {description}")


def project_transactions() -> list[Path]:
    return sorted(
        path
        for path in ROOT.glob(".lligma-mutation-*")
        if path.is_dir() and not path.is_symlink()
    )


def kill_pid(pid: int) -> None:
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def kill_tree(pid: int) -> None:
    """Uccide il processo e i suoi figli, senza toccare il watchdog WMI."""
    if pid <= 0:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        kill_pid(pid)


def read_log(log) -> str:
    log.flush()
    log.seek(0)
    return log.read()


def assert_mutation_active() -> None:
    text = SERVER.read_text(encoding="utf-8")
    lock = "        with server.stream_aborts_lock:\n"
    mutation = '        server.stream_aborts = getattr(server, "stream_aborts", 0) + 1\n'
    if lock in text or mutation not in text:
        raise AssertionError("server.py non e' nello stato mutato atteso")


def interrupt_full_command(before: tuple[str, str]) -> None:
    watchdog_pid = 0
    with tempfile.TemporaryDirectory(prefix="lligma-interrupt-") as directory:
        marker = Path(directory) / "mutation-applied"
        log_path = Path(directory) / "command.log"
        env = os.environ.copy()
        env["LLIGMA_MUTATION_PAUSE_AFTER"] = "server"
        env["LLIGMA_MUTATION_PAUSE_FILE"] = str(marker)
        with log_path.open("w+", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                [sys.executable, str(MUTATION)],
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=(os.name != "nt"),
            )
            try:
                wait_until(marker.exists, 120, "mutazione server applicata")
                assert_mutation_active()
                transactions = project_transactions()
                assert len(transactions) == 1, transactions
                transaction = transactions[0]
                watchdog_pid = int(
                    (transaction / "watchdog.pid").read_text(encoding="ascii").strip()
                )
                assert process_alive(watchdog_pid), "watchdog non attivo"
                print(f"INTERRUPT: mutazione applicata, transazione {transaction.name}")
                print(f"INTERRUPT: watchdog pid {watchdog_pid}, kill dell'albero del comando")
                kill_tree(process.pid)
                process.wait(timeout=20)
                # Se il watchdog e' un figlio normale, taskkill /T lo uccide:
                # non aspettare il timeout di ripristino, perche' il FAIL deve
                # attribuire la causa alla mancata sopravvivenza out-of-tree.
                restore_deadline = time.monotonic() + 30
                while time.monotonic() < restore_deadline:
                    restored = hashes() == before and not transaction.exists()
                    watchdog_alive = process_alive(watchdog_pid)
                    if restored and not watchdog_alive:
                        break
                    if not watchdog_alive:
                        raise AssertionError(
                            "watchdog non sopravvissuto a taskkill /T: "
                            f"pid {watchdog_pid} morto prima del ripristino "
                            f"(transazione={transaction.name}, "
                            f"hash_restored={hashes() == before})"
                        )
                    time.sleep(0.1)
                else:
                    raise AssertionError("timeout: ripristino del watchdog")
            finally:
                if process.poll() is None:
                    kill_tree(process.pid)
                    process.wait(timeout=20)
                output = read_log(log)
            if "Traceback" in output:
                print(output)
            print("WATCHDOG_RESTORE: PASS")
            print(f"HASHES_RESTORED: {hashes() == before}")
            print(f"BACKUP_REMOVED: {not project_transactions()}")
            print(f"WATCHDOG_STOPPED: {not process_alive(watchdog_pid)}")


def make_orphan_transaction() -> Path:
    with tempfile.TemporaryDirectory(prefix="lligma-orphan-") as directory:
        marker = Path(directory) / "txn-ready.json"
        log_path = Path(directory) / "child.log"
        code = r"""
import json, os, sys, time
from pathlib import Path
sys.path.insert(0, 'tests')
from mutation_transaction import MutationTransaction
from mutation_fault_harness import apply_mutation, lock_blocks
server = Path('lligma/tutor/server.py')
text = server.read_text(encoding='utf-8')
lock, mutated = lock_blocks(text)
txn = MutationTransaction.begin(server, Path('lligma/tutor/service.py'))
apply_mutation(server, text, lock, mutated)
Path(os.environ['ORPHAN_MARKER']).write_text(
    json.dumps({'transaction': str(txn.path), 'watchdog': txn.watchdog.pid}),
    encoding='utf-8',
)
print('orphan-ready', flush=True)
time.sleep(120)
"""
        env = os.environ.copy()
        env["ORPHAN_MARKER"] = str(marker)
        env["PYTHONPATH"] = "tests"
        with log_path.open("w+", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                [sys.executable, "-c", code],
                cwd=str(ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            try:
                wait_until(marker.exists, 30, "transazione orfana pronta")
                payload = json.loads(marker.read_text(encoding="utf-8"))
                transaction = Path(payload["transaction"])
                watchdog_pid = int(payload["watchdog"])
                assert transaction.is_dir(), transaction
                assert process_alive(watchdog_pid), "watchdog orfano non attivo"
                print(f"ORPHAN: creata {transaction.name}, watchdog {watchdog_pid}")
                # Uccidere prima il guardiano, poi il padre: resta il caso peggiore,
                # con file mutati e backup presenti all'avvio successivo.
                kill_pid(watchdog_pid)
                process.kill()
                process.wait(timeout=20)
                wait_until(
                    lambda: not process_alive(watchdog_pid),
                    10,
                    "watchdog orfano terminato",
                )
                assert transaction.is_dir(), "la transazione orfana e' sparita troppo presto"
                assert_mutation_active()
                print("ORPHAN: file mutato e backup lasciati di proposito")
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=20)
                output = read_log(log)
            if "Traceback" in output:
                print(output)
        return transaction


def run_full_after_orphan(before: tuple[str, str], transaction: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(MUTATION)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=240,
    )
    output = (result.stdout or "") + (result.stderr or "")
    print(output[-12000:])
    assert result.returncode == 0, f"comando completo exit {result.returncode}"
    assert "RECOVERY:" in output, "il comando non ha segnalato il recovery"
    assert "VERDICT: PASS" in output, "il comando completo non ha dato PASS"
    assert "mutated: 3/3 run fallite" in output, output
    assert "cache-mutated: 3/3 run fallite" in output, output
    assert "transaction_cleanup_ok=True" in output, output
    assert hashes() == before, (hashes(), before)
    assert not transaction.exists(), transaction
    assert not project_transactions(), project_transactions()
    print("FULL_AFTER_RECOVERY: PASS")


def main() -> int:
    before = hashes()
    recovered, errors = recover_pending()
    for message in recovered:
        print(f"PRE_RECOVERY: {message}")
    if errors:
        for message in errors:
            print(f"ERRORE: {message}")
        return 2
    if hashes() != before:
        print("ERRORE: hash cambiati durante il recovery iniziale")
        return 2
    print(f"SNAPSHOT: server={before[0]} service={before[1]}")
    interrupt_full_command(before)
    orphan = make_orphan_transaction()
    run_full_after_orphan(before, orphan)
    print("INTERRUPTION_DEMO: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
