#!/usr/bin/env python3
"""Controllo negativo automatico del safety harness.

Il test e' eseguibile sia direttamente sia da pytest::

    python tests/test_mutation_negative_control.py
    python -m pytest tests/test_mutation_negative_control.py -q

Ogni caso salva in memoria i byte dei file coinvolti, applica una sola
regressione a ``mutation_transaction.py``, esegue il safety e verifica il FAIL
atteso. Il ripristino avviene sempre in ``finally``, anche su eccezioni o
interruzioni, e viene verificato contro gli hash congelati.

Con il guard out-of-tree sabotato il comando non protetto puo' uscire 0 o 1 a
seconda che la suite E2E concorrente passi o fallisca sotto carico, oppure
restare in esecuzione fino al timeout di 240 s del safety. Il test accetta
questi esiti, perche' il contratto del guard e' soltanto "non exit 2 prima
della mutazione"; il safety deve comunque fallire e riportare l'esito reale.

Nessun backup o log viene scritto nel progetto: lo snapshot e' in memoria e
l'output del sottoprocesso e' catturato. L'unico file temporaneo e' il canary
del test di sabotaggio, creato in una directory temporanea del sistema.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements.txt"
SERVER = ROOT / "lligma" / "tutor" / "server.py"
SERVICE = ROOT / "lligma" / "tutor" / "service.py"
SAFETY = ROOT / "tests" / "mutation_safety.py"
INTERRUPTION = ROOT / "tests" / "mutation_interruption.py"
TRANSACTION = ROOT / "tests" / "mutation_transaction.py"
WATCHDOG = ROOT / "tests" / "mutation_watchdog.py"

FROZEN_MANIFEST = ROOT / "tests" / "frozen_manifest.json"

FROZEN = {
    "requirements.txt": "96827e935664b43f65f14142d1e6407b1226fd69389401ae5716f2dace1dc5e6",
    "lligma/tutor/server.py": "53cde9d799e13829710b99a0a624bac01150a22b86d0d5721a833f2222705e69",
    "lligma/tutor/service.py": "ba869e26806718ab7a856f05994ab37248d73b91d6e729adf4b2fe681badaa53",
    "tests/mutation_safety.py": "b3c8d67a0429d52b13b0bcac849e5e726209831c777ba3d80c711a8b934e5c17",
    "tests/mutation_interruption.py": "ab5a96e8a112ac9c61461661554eb8f665fd19f756ad938f71997c5d5bc740e4",
    "tests/mutation_transaction.py": "8bce56baba0b7d5220dac201ada33c15aede3e5ba4ad673b63d42b0f9c5d7ea4",
    "tests/mutation_watchdog.py": "e215845280ff6716102ef696aee803ad2be874971788ea7197198129c6172df4",
}

TREE_OLD = """        if os.name == \"nt\":
            return _start_wmi_watchdog(transaction, root)
"""
TREE_NEW = """        if os.name == \"nt\":
            # REGRESSIONE TEMPORANEA: watchdog figlio normale, non out-of-tree.
            env = os.environ.copy()
            env.pop(\"LLIGMA_MUTATION_PAUSE_AFTER\", None)
            env.pop(\"LLIGMA_MUTATION_PAUSE_FILE\", None)
            command = [sys.executable, str(WATCHDOG), \"--transaction\", str(transaction)]
            process = subprocess.Popen(
                command,
                cwd=str(root),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
            )
            return WatchdogHandle(pid=process.pid, process=process)
"""

GUARD_OLD = 'if os.environ.get(OUT_OF_TREE_DISABLE_ENV) == "1":\n'
GUARD_NEW = 'if os.environ.get(OUT_OF_TREE_DISABLE_ENV) == "0":\n'

CASE_TIMEOUT_SECONDS = 300
KILL_TIMEOUT_SECONDS = 20
OUTPUT_LINE_TOKENS = (
    "TREE_KILL",
    "OUT_OF_TREE",
    "MUTATION_SAFETY",
    "WATCHDOG_RESTORE",
    "CLEANUP",
    "FAIL",
    "ERROR",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_frozen() -> dict[str, str]:
    payload = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in payload.items()
    ):
        raise AssertionError(f"{FROZEN_MANIFEST.name}: manifest non valido")
    return payload


def assert_frozen(label: str) -> None:
    canonical = canonical_frozen()
    ledger_wrong = []
    if set(canonical) != set(FROZEN):
        ledger_wrong.append(
            f"chiavi: FROZEN={sorted(FROZEN)} manifest={sorted(canonical)}"
        )
    for relative, expected in FROZEN.items():
        if canonical.get(relative) != expected:
            ledger_wrong.append(
                f"{relative}: FROZEN={expected} manifest={canonical.get(relative)}"
            )
    if ledger_wrong:
        raise AssertionError(
            f"{label}: FROZEN non coincide con {FROZEN_MANIFEST.name}: "
            + "; ".join(ledger_wrong)
        )
    wrong = []
    for relative, expected in FROZEN.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            wrong.append(f"{relative}: {actual} != {expected}")
    if wrong:
        raise AssertionError(f"{label}: hash congelati diversi: " + "; ".join(wrong))


def snapshot() -> dict[Path, bytes]:
    return {ROOT / relative: (ROOT / relative).read_bytes() for relative in FROZEN}


def restore_snapshot(files: dict[Path, bytes], *, apply: bool = True) -> None:
    """Ripristina e verifica; con ``apply=False`` simula un ripristino disattivato."""
    if apply:
        for path, data in files.items():
            path.write_bytes(data)
    wrong = [
        f"{path}: {sha256(path)} != {hashlib.sha256(data).hexdigest()}"
        for path, data in files.items()
        if sha256(path) != hashlib.sha256(data).hexdigest()
    ]
    if wrong:
        raise AssertionError("ripristino non verificato: " + "; ".join(wrong))


def project_artifacts() -> list[Path]:
    artifacts = list(ROOT.glob(".lligma-mutation-*"))
    artifacts.extend(ROOT.glob(".lligma-recovery-*"))
    artifacts.extend(ROOT.glob(".negative-control-*"))
    return sorted(artifacts)


def cleanup_artifacts() -> None:
    for path in ROOT.glob(".lligma-mutation-*"):
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except OSError:
                pass
    for path in ROOT.glob(".lligma-recovery-*"):
        try:
            path.unlink()
        except OSError:
            pass


def assert_no_artifacts(label: str) -> None:
    artifacts = project_artifacts()
    if artifacts:
        raise AssertionError(f"{label}: residui nel progetto: {artifacts}")


def kill_tree(pid: int) -> None:
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
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_safety() -> tuple[int, str]:
    env = os.environ.copy()
    env.pop("LLIGMA_MUTATION_DISABLE_OUT_OF_TREE", None)
    process = subprocess.Popen(
        [sys.executable, "-u", str(SAFETY)],
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        start_new_session=(os.name != "nt"),
    )
    try:
        try:
            output, _ = process.communicate(timeout=CASE_TIMEOUT_SECONDS)
            return process.returncode, output or ""
        except subprocess.TimeoutExpired:
            kill_tree(process.pid)
            output, _ = process.communicate(timeout=KILL_TIMEOUT_SECONDS)
            return 124, (output or "") + "\nNEGATIVE_CONTROL_TIMEOUT\n"
    finally:
        if process.poll() is None:
            kill_tree(process.pid)
            try:
                process.wait(timeout=KILL_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()


def print_relevant(name: str, output: str) -> None:
    print(f"{name}_OUTPUT:")
    for line in output.splitlines():
        if any(token in line for token in OUTPUT_LINE_TOKENS):
            print(f"  {line}")


def run_case(name: str, old: str, new: str, expected: str) -> None:
    assert_frozen(f"{name} before")
    original = snapshot()
    source = original[TRANSACTION]
    old_bytes = old.encode("utf-8")
    new_bytes = new.encode("utf-8")
    if source.count(old_bytes) != 1:
        raise AssertionError(f"{name}: ancora di patch trovata {source.count(old_bytes)} volte")

    code = 0
    output = ""
    try:
        # Patch byte-to-byte: non normalizzare i line ending del file originale.
        TRANSACTION.write_bytes(source.replace(old_bytes, new_bytes, 1))
        code, output = run_safety()
    finally:
        # Deve avvenire anche se il safety va in timeout o arriva KeyboardInterrupt.
        restore_snapshot(original)
        cleanup_artifacts()
        assert_frozen(f"{name} after restore")
        assert_no_artifacts(name)

    print_relevant(name, output)
    if code != 1:
        raise AssertionError(f"{name}: atteso exit 1, ottenuto {code}")
    if expected not in output:
        raise AssertionError(f"{name}: messaggio atteso assente: {expected!r}")
    if "MUTATION_SAFETY: FAIL" not in output:
        raise AssertionError(f"{name}: manca MUTATION_SAFETY: FAIL")
    if name == "TREE_KILL":
        if "timeout: ripristino del watchdog" in output:
            raise AssertionError(f"{name}: FAIL dovuto al timeout di scenario")
        if "Traceback (most recent call last)" in output:
            raise AssertionError(f"{name}: FAIL con traceback di orchestrazione")
    print(f"{name}: PASS")


def run_all() -> None:
    run_case(
        "TREE_KILL",
        TREE_OLD,
        TREE_NEW,
        "watchdog non sopravvissuto a taskkill /T",
    )
    run_case(
        "OUT_OF_TREE_UNAVAILABLE",
        GUARD_OLD,
        GUARD_NEW,
        # L'exit del comando non protetto dipende anche dalla suite E2E, che
        # sotto carico puo' fallire prima della mutazione o restare in
        # esecuzione fino al timeout. Il safety deve fallire in ogni caso e
        # riportare che il guard non ha dato 2.
        "out-of-tree assente: atteso exit 2, ottenuto",
    )
    assert_frozen("negative control complete")
    assert_no_artifacts("negative control complete")


def assert_ledger_writer_refuses() -> None:
    writer = ROOT / "outputs" / "frozen_ledger.py"
    with tempfile.TemporaryDirectory(prefix="lligma-ledger-writer-") as directory:
        root = Path(directory)
        (root / "tests").mkdir()
        control = root / "tests" / "test_mutation_negative_control.py"
        manifest = root / "tests" / "frozen_manifest.json"
        frozen = root / "frozen.txt"
        control.write_text('FROZEN = {"frozen.txt": "' + "0" * 64 + '"}', encoding="utf-8")
        manifest.write_text(json.dumps({"frozen.txt": "1" * 64}), encoding="utf-8")
        frozen.write_text("mutated", encoding="utf-8")
        before = {path: path.read_bytes() for path in (control, manifest, frozen)}
        result = subprocess.run(
            [sys.executable, str(writer), "--root", str(root), "--realign-from-disk"],
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        after = {path: path.read_bytes() for path in (control, manifest, frozen)}
        if result.returncode != 3:
            raise AssertionError(
                f"ledger writer: atteso exit 3, ottenuto {result.returncode}: {result.stdout}"
            )
        if "LEDGER_REALIGNMENT_REFUSED" not in result.stdout:
            raise AssertionError(f"ledger writer: messaggio assente: {result.stdout!r}")
        changed = [str(path) for path in before if before[path] != after[path]]
        if changed:
            raise AssertionError(f"ledger writer: ha scritto: {changed}")
        print("LEDGER_WRITER_REFUSAL: PASS")


def assert_ledger_files_read_only() -> None:
    if os.name != "nt":
        return
    for path in (Path(__file__).resolve(), FROZEN_MANIFEST):
        try:
            with path.open("r+b"):
                pass
        except PermissionError:
            continue
        raise AssertionError(
            f"{path}: scrivibile; la protezione write-point del ledger non e' attiva"
        )
    print("LEDGER_WRITE_POINT_READONLY: PASS")


def test_negative_control_automated() -> None:
    run_all()


def test_restore_guard_detects_sabotage() -> None:
    assert_ledger_writer_refuses()
    assert_ledger_files_read_only()
    with tempfile.TemporaryDirectory(prefix="lligma-negative-control-sabotage-") as directory:
        canary = Path(directory) / "canary.txt"
        original = b"original\n"
        canary.write_bytes(original)
        files = {canary: original}
        canary.write_bytes(b"mutated\n")
        try:
            try:
                restore_snapshot(files, apply=False)
            except AssertionError as error:
                if "ripristino non verificato" not in str(error):
                    raise
                print("SABOTAGE_RESTORE_GUARD: PASS")
                return
            raise AssertionError("ripristino disattivato non rilevato")
        finally:
            canary.write_bytes(original)


def main() -> int:
    try:
        run_all()
        test_restore_guard_detects_sabotage()
    except BaseException as error:
        print(f"NEGATIVE_CONTROL: FAIL: {type(error).__name__}: {error}", flush=True)
        return 1
    print("NEGATIVE_CONTROL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
