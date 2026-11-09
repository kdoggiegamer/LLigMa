#!/usr/bin/env python3
"""Guardiano esterno del comando di mutazione.

Il processo principale crea una transazione con i backup e avvia questo
processo separato. Il guardiano aspetta che il padre termini; se non trova un
marker ``clean.json`` valido, ripristina i file dai backup e verifica gli hash
originali. Se il padre viene terminato con un kill non intercettabile, il
ripristino avviene comunque qui, fuori dal ``finally`` del padre. Il padre e'
identificato anche dall'ora di creazione, cosi' un PID riutilizzato non tiene
il guardiano in attesa per sempre.

Su Windows il processo viene creato da WMI, fuori dall'albero del comando, cosi'
un ``taskkill /T`` non lo uccide; su POSIX gira in una nuova sessione. Non
importa moduli del progetto: deve poter lavorare anche se il codice sotto
mutazione e' in uno stato non importabile.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

MANIFEST = "manifest.json"
CLEAN = "clean.json"
READY = "ready"
WATCHDOG_PID = "watchdog.pid"
ERROR = "watchdog-error.txt"
POLL_SECONDS = 0.1
GRACE_SECONDS = 0.25


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def write_json(path: Path, payload: dict) -> None:
    atomic_write(
        path,
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"),
    )


def process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        synchronize = 0x00100000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(synchronize, False, int(pid))
        if not handle:
            return False
        try:
            return kernel32.WaitForSingleObject(handle, 0) == 0x00000102
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_start_time(pid: int) -> int | None:
    """Identita' del padre oltre il PID, contro il riuso del PID."""
    if pid <= 0:
        return None
    if os.name == "nt":
        class FILETIME(ctypes.Structure):
            _fields_ = [
                ("dwLowDateTime", ctypes.c_uint32),
                ("dwHighDateTime", ctypes.c_uint32),
            ]

        query_limited = 0x1000
        query = 0x0400
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(query_limited, False, int(pid))
        if not handle:
            handle = kernel32.OpenProcess(query, False, int(pid))
        if not handle:
            return None
        try:
            created = FILETIME()
            exited = FILETIME()
            kernel = FILETIME()
            user = FILETIME()
            ok = kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            )
            if not ok:
                return None
            return (int(created.dwHighDateTime) << 32) | int(created.dwLowDateTime)
        finally:
            kernel32.CloseHandle(handle)
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        fields = stat.rsplit(")", 1)[1].split()
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _same_process(pid: int, expected_start: int | None) -> bool:
    if not process_alive(pid):
        return False
    if expected_start is None:
        return True
    actual = process_start_time(pid)
    return actual is None or actual == expected_start


def _target_path(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    root_resolved = root.resolve()
    if target != root_resolved and root_resolved not in target.parents:
        raise ValueError(f"percorso di ripristino fuori dal progetto: {relative!r}")
    return target


def _files_match(root: Path, files: list[dict]) -> bool:
    for entry in files:
        target = _target_path(root, entry["path"])
        if not target.is_file() or sha256(target.read_bytes()) != entry["sha256"]:
            return False
    return True


def _restore(root: Path, transaction: Path, files: list[dict]) -> str | None:
    try:
        prepared: list[tuple[Path, bytes]] = []
        for entry in files:
            target = _target_path(root, entry["path"])
            backup = transaction / entry["backup"]
            if not backup.is_file():
                return f"backup mancante per {entry['path']}: {backup}"
            data = backup.read_bytes()
            expected = entry["sha256"]
            if sha256(data) != expected:
                return f"backup corrotto per {entry['path']}: hash diverso"
            prepared.append((target, data))
        for target, data in prepared:
            atomic_write(target, data)
            if sha256(target.read_bytes()) != sha256(data):
                return f"ripristino non verificato per {target.name}"
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


def _remove(transaction: Path) -> str | None:
    try:
        shutil.rmtree(transaction)
    except Exception as error:
        return f"rimozione transazione fallita: {type(error).__name__}: {error}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transaction", required=True)
    args = parser.parse_args()

    transaction = Path(args.transaction).resolve()
    root = transaction.parent
    # Il padre puo' essere interrotto con Ctrl+C: il guardiano non deve
    # ereditare l'interruzione mentre sta ripristinando.
    for name in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, name):
            try:
                signal.signal(getattr(signal, name), signal.SIG_IGN)
            except (OSError, ValueError):
                pass

    manifest_path = transaction / MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
        parent_pid = int(manifest["parent_pid"])
        parent_start = manifest.get("parent_start")
        if parent_start is not None and not isinstance(parent_start, int):
            raise ValueError("parent_start non valido")
        if not isinstance(files, list) or not files:
            raise ValueError("manifest senza file")
        if not parent_pid:
            raise ValueError("parent_pid non valido")
    except Exception as error:
        print(f"WATCHDOG ERROR: manifest non valido: {type(error).__name__}: {error}", file=sys.stderr)
        return 2

    atomic_write(transaction / READY, b"ready\n")

    while _same_process(parent_pid, parent_start):
        if not transaction.exists():
            # Il padre ha gia' rimosso la transazione dopo il clean marker.
            return 0
        if (transaction / CLEAN).is_file():
            break
        time.sleep(POLL_SECONDS)

    # Il padre puo' aver scritto clean.json poco prima di uscire.
    time.sleep(GRACE_SECONDS)
    clean_path = transaction / CLEAN
    if clean_path.is_file():
        try:
            clean = json.loads(clean_path.read_text(encoding="utf-8"))
            clean_ok = (
                clean.get("status") == "clean"
                and isinstance(clean.get("files"), list)
                and _files_match(root, clean["files"])
                and _files_match(root, files)
            )
        except Exception:
            clean_ok = False
        if clean_ok:
            error = _remove(transaction)
            if error is None:
                return 0
            print(f"WATCHDOG ERROR: {error}", file=sys.stderr)
            return 1

    error = _restore(root, transaction, files)
    if error is not None:
        try:
            write_json(transaction / ERROR, {"error": error})
        except OSError:
            pass
        print(f"WATCHDOG ERROR: {error}", file=sys.stderr)
        return 1

    error = _remove(transaction)
    if error is not None:
        print(f"WATCHDOG ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
