"""Transazione e recovery per il comando di mutazione.

La transazione contiene i byte originali di ``server.py`` e ``service.py`` e un
manifest con gli hash. Il comando principale la crea prima di applicare una
mutazione e avvia ``mutation_watchdog.py`` fuori dall'albero del processo: su
Windows usa ``Win32_Process.Create`` via WMI, cosi' un ``taskkill /T`` del
comando non puo' uccidere anche il guardiano; su POSIX usa una nuova sessione.
Il guardiano ripristina i file se il padre muore senza un marker clean valido.

All'avvio ``recover_pending`` cerca transazioni rimaste da esecuzioni
precedenti: se il padre non e' piu' vivo, ripristina dai backup e verifica gli
hash. Una transazione con padre ancora vivo viene rifiutata, non cancellata.
Il padre viene identificato con PID e ora di creazione, per non confondere un
PID riutilizzato. Il recovery usa un lock consultivo per transazione e un
preflight di tutti i backup prima di scrivere: non lascia un ripristino
parziale se un backup manca o e' corrotto, e non lascia watchdog vivi dopo il
clean.
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
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = ".lligma-mutation-"
MANIFEST = "manifest.json"
CLEAN = "clean.json"
READY = "ready"
WATCHDOG_PID = "watchdog.pid"
ERROR = "watchdog-error.txt"
RECOVERY_LOCK_PREFIX = ".lligma-recovery-"
WATCHDOG = Path(__file__).with_name("mutation_watchdog.py")
START_TIMEOUT_SECONDS = 10.0
STOP_TIMEOUT_SECONDS = 10.0
RECOVERY_WAIT_SECONDS = 10.0
POLL_SECONDS = 0.05
OUT_OF_TREE_DISABLE_ENV = "LLIGMA_MUTATION_DISABLE_OUT_OF_TREE"


@dataclass
class WatchdogHandle:
    """Handle uniforme per un watchdog Popen o creato da WMI."""

    pid: int
    process: subprocess.Popen | None = None

    def poll(self) -> int | None:
        if self.process is not None:
            return self.process.poll()
        return None if process_alive(self.pid) else 0

    def terminate(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
            return
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(self.pid), "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        try:
            os.kill(self.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def kill(self) -> None:
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            return
        self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        if self.process is not None:
            return self.process.wait(timeout=timeout)
        deadline = None if timeout is None else time.monotonic() + timeout
        while process_alive(self.pid):
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("watchdog", timeout or 0)
            time.sleep(POLL_SECONDS)
        return 0


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
        import ctypes

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
    """Identita' del processo oltre il PID, per non confondere un PID riusato."""
    if pid <= 0:
        return None
    if os.name == "nt":
        import ctypes

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
        # Manifest precedente senza identita': conserva il comportamento sicuro.
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


def _read_manifest(transaction: Path) -> dict:
    payload = json.loads((transaction / MANIFEST).read_text(encoding="utf-8"))
    files = payload.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest senza file")
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("voce manifest non valida")
        for key in ("path", "backup", "sha256"):
            if not isinstance(entry.get(key), str) or not entry[key]:
                raise ValueError(f"campo manifest mancante: {key}")
    if not isinstance(payload.get("parent_pid"), int):
        raise ValueError("parent_pid mancante")
    if "parent_start" in payload and not isinstance(payload["parent_start"], int):
        raise ValueError("parent_start non valido")
    return payload


def _wait_for_path(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(POLL_SECONDS)
    return path.exists()


def _restore_from_transaction(transaction: Path, files: list[dict]) -> str | None:
    root = transaction.parent
    try:
        # Preflight: nessun file viene toccato finche' tutti i backup non sono
        # completi e coerenti. Evita ripristini parziali con due file.
        prepared: list[tuple[Path, bytes]] = []
        for entry in files:
            target = _target_path(root, entry["path"])
            backup = transaction / entry["backup"]
            if not backup.is_file():
                return f"backup mancante per {entry['path']}: {backup}"
            data = backup.read_bytes()
            if sha256(data) != entry["sha256"]:
                return f"backup corrotto per {entry['path']}"
            prepared.append((target, data))
        for target, data in prepared:
            atomic_write(target, data)
            if sha256(target.read_bytes()) != sha256(data):
                return f"ripristino non verificato per {target.name}"
    except Exception as error:
        return f"{type(error).__name__}: {error}"
    return None


class RecoveryLock:
    """Lock consultivo per impedire due recovery simultanei della stessa txn."""

    def __init__(self, path: Path):
        self.path = path
        self.handle = None
        self.acquired = False

    def acquire(self) -> bool:
        try:
            self.path.touch(exist_ok=True)
            self.handle = self.path.open("r+b")
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.acquired = True
            return True
        except OSError:
            if self.handle is not None:
                self.handle.close()
                self.handle = None
            return False

    def release(self) -> None:
        if self.handle is not None:
            if self.acquired:
                try:
                    self.handle.seek(0)
                    if os.name == "nt":
                        import msvcrt

                        msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            self.handle.close()
            self.handle = None
        owned = self.acquired
        self.acquired = False
        if owned:
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass


def _recovery_lock_path(root: Path, transaction: Path) -> Path:
    return root / f"{RECOVERY_LOCK_PREFIX}{transaction.name}.lock"


def _wait_pid_gone(pid: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while process_alive(pid):
        if time.monotonic() >= deadline:
            return False
        time.sleep(POLL_SECONDS)
    return True


def _remove_transaction(transaction: Path) -> str | None:
    if not transaction.exists():
        return None
    try:
        shutil.rmtree(transaction)
    except Exception as error:
        return f"rimozione transazione fallita: {type(error).__name__}: {error}"
    return None


@dataclass
class MutationTransaction:
    path: Path
    manifest: dict
    watchdog: WatchdogHandle

    @property
    def server_backup(self) -> Path:
        return self.path / "server.py"

    @property
    def service_backup(self) -> Path:
        return self.path / "service.py"

    @classmethod
    def begin(cls, server: Path, service: Path, root: Path = ROOT) -> "MutationTransaction":
        root = root.resolve()
        server = server.resolve()
        service = service.resolve()
        transaction = Path(tempfile.mkdtemp(prefix=PREFIX, dir=root))
        watchdog: WatchdogHandle | None = None
        try:
            server_bytes = server.read_bytes()
            service_bytes = service.read_bytes()
            (transaction / "server.py").write_bytes(server_bytes)
            (transaction / "service.py").write_bytes(service_bytes)
            files = []
            for relative, data in (
                (server.relative_to(root), server_bytes),
                (service.relative_to(root), service_bytes),
            ):
                files.append(
                    {
                        "path": relative.as_posix(),
                        "backup": relative.name,
                        "sha256": sha256(data),
                    }
                )
            manifest = {
                "version": 1,
                "parent_pid": os.getpid(),
                "files": files,
            }
            parent_start = process_start_time(os.getpid())
            if parent_start is not None:
                manifest["parent_start"] = parent_start
            write_json(transaction / MANIFEST, manifest)
            watchdog = cls._start_watchdog(transaction, root)
            if not _wait_for_path(transaction / READY, START_TIMEOUT_SECONDS):
                raise RuntimeError("watchdog non pronto")
            (transaction / WATCHDOG_PID).write_text(str(watchdog.pid), encoding="ascii")
            return cls(path=transaction, manifest=manifest, watchdog=watchdog)
        except BaseException:
            if watchdog is not None and watchdog.poll() is None:
                watchdog.terminate()
                try:
                    watchdog.wait(timeout=STOP_TIMEOUT_SECONDS)
                except subprocess.TimeoutExpired:
                    watchdog.kill()
            shutil.rmtree(transaction, ignore_errors=True)
            raise

    @staticmethod
    def _start_watchdog(transaction: Path, root: Path) -> WatchdogHandle:
        if os.environ.get(OUT_OF_TREE_DISABLE_ENV) == "1":
            raise RuntimeError(
                f"meccanismo out-of-tree disabilitato da {OUT_OF_TREE_DISABLE_ENV}"
            )
        if os.name == "nt":
            return _start_wmi_watchdog(transaction, root)

        env = os.environ.copy()
        env.pop("LLIGMA_MUTATION_PAUSE_AFTER", None)
        env.pop("LLIGMA_MUTATION_PAUSE_FILE", None)
        command = [sys.executable, str(WATCHDOG), "--transaction", str(transaction)]
        process = subprocess.Popen(
            command,
            cwd=str(root),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
        return WatchdogHandle(pid=process.pid, process=process)

    def finish(self, restored: bool) -> bool:
        """Scrive il marker clean solo dopo un ripristino verificato."""
        if not restored or not _files_match(self.path.parent, self.manifest["files"]):
            return False
        write_json(
            self.path / CLEAN,
            {"status": "clean", "files": self.manifest["files"]},
        )
        self._stop_watchdog()
        if self.watchdog.poll() is None:
            return False
        removed = _remove_transaction(self.path)
        return removed is None

    def _stop_watchdog(self) -> None:
        if self.watchdog.poll() is not None:
            return
        self.watchdog.terminate()
        try:
            self.watchdog.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            self.watchdog.kill()
            try:
                self.watchdog.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                pass


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _start_wmi_watchdog(transaction: Path, root: Path) -> WatchdogHandle:
    """Crea il watchdog tramite WMI, fuori dall'albero del processo padre."""
    executable = shutil.which("powershell.exe") or shutil.which("powershell")
    if not executable:
        raise RuntimeError(
            "watchdog Windows non avviabile: powershell.exe non trovato per Win32_Process.Create"
        )
    command = [sys.executable, str(WATCHDOG), "--transaction", str(transaction)]
    command_line = subprocess.list2cmdline(command)
    script = "\n".join(
        [
            "$ErrorActionPreference = 'Stop'",
            f"$cmd = {_powershell_quote(command_line)}",
            "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create "
            f"-Arguments @{{CommandLine = $cmd; CurrentDirectory = {_powershell_quote(str(root))}}}",
            "if ($null -eq $r -or $r.ReturnValue -ne 0) {",
            "  Write-Output ('WMI_RETURN=' + $r.ReturnValue); exit 1",
            "}",
            "Write-Output $r.ProcessId",
        ]
    )
    try:
        completed = subprocess.run(
            [
                executable,
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=START_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError(f"avvio watchdog WMI fallito: {type(error).__name__}: {error}") from error
    if completed.returncode != 0:
        detail = (completed.stdout or "") + (completed.stderr or "")
        raise RuntimeError(f"avvio watchdog WMI fallito: {detail.strip() or completed.returncode}")
    pid_lines = [
        line.strip()
        for line in (completed.stdout or "").splitlines()
        if line.strip().isdigit()
    ]
    if not pid_lines:
        raise RuntimeError(f"WMI non ha restituito un PID: {(completed.stdout or '').strip()!r}")
    pid = int(pid_lines[-1])
    if pid <= 0 or not process_alive(pid):
        raise RuntimeError(f"watchdog WMI non attivo dopo la creazione (pid {pid})")
    return WatchdogHandle(pid=pid)


def _recover_one(
    transaction: Path,
    manifest: dict,
    root: Path,
    recovered: list[str],
    errors: list[str],
) -> None:
    watchdog_pid = 0
    try:
        watchdog_pid = int((transaction / WATCHDOG_PID).read_text(encoding="ascii").strip())
    except (OSError, ValueError):
        pass
    deadline = time.monotonic() + RECOVERY_WAIT_SECONDS
    while watchdog_pid and process_alive(watchdog_pid) and time.monotonic() < deadline:
        time.sleep(POLL_SECONDS)
    if not transaction.exists():
        if watchdog_pid and not _wait_pid_gone(watchdog_pid, RECOVERY_WAIT_SECONDS):
            errors.append(
                f"transazione {transaction.name}: watchdog {watchdog_pid} ancora vivo "
                "dopo il recovery"
            )
        else:
            recovered.append(f"{transaction.name}: ripristinata dal watchdog")
        return

    clean_ok = False
    clean_path = transaction / CLEAN
    if clean_path.is_file():
        try:
            clean = json.loads(clean_path.read_text(encoding="utf-8"))
            clean_ok = clean.get("status") == "clean" and _files_match(
                root, clean.get("files", [])
            )
        except Exception:
            clean_ok = False
    if clean_ok and _files_match(root, manifest["files"]):
        error = _remove_transaction(transaction)
        if error is not None:
            errors.append(error)
            return
        if watchdog_pid and not _wait_pid_gone(watchdog_pid, RECOVERY_WAIT_SECONDS):
            errors.append(
                f"transazione {transaction.name}: watchdog {watchdog_pid} ancora vivo "
                "dopo il clean"
            )
            return
        recovered.append(f"{transaction.name}: stato clean verificato")
        return

    error = _restore_from_transaction(transaction, manifest["files"])
    if error is not None:
        errors.append(f"{transaction.name}: {error}")
        return
    error = _remove_transaction(transaction)
    if error is not None:
        errors.append(error)
        return
    if watchdog_pid and not _wait_pid_gone(watchdog_pid, RECOVERY_WAIT_SECONDS):
        errors.append(
            f"transazione {transaction.name}: watchdog {watchdog_pid} ancora vivo "
            "dopo il ripristino"
        )
        return
    recovered.append(f"{transaction.name}: ripristinata da backup")


def _cleanup_recovery_locks(root: Path) -> None:
    for path in root.glob(RECOVERY_LOCK_PREFIX + "*.lock"):
        lock = RecoveryLock(path)
        if lock.acquire():
            lock.release()


def recover_pending(root: Path = ROOT) -> tuple[list[str], list[str]]:
    """Ripristina transazioni orfane, con lock contro recovery concorrenti."""
    root = root.resolve()
    _cleanup_recovery_locks(root)
    recovered: list[str] = []
    errors: list[str] = []
    for transaction in sorted(root.glob(PREFIX + "*")):
        if not transaction.is_dir() or transaction.is_symlink():
            continue
        try:
            manifest = _read_manifest(transaction)
        except Exception as error:
            errors.append(
                f"transazione {transaction.name} illeggibile: {type(error).__name__}: {error}"
            )
            continue

        parent_pid = int(manifest["parent_pid"])
        parent_start = manifest.get("parent_start")
        if _same_process(parent_pid, parent_start):
            errors.append(
                f"transazione {transaction.name} ancora attiva (pid {parent_pid}); "
                "attendere o terminare quel processo"
            )
            continue
        if process_alive(parent_pid):
            print(
                f"RECOVERY: {transaction.name}: pid {parent_pid} riutilizzato; "
                "il padre originale non e' piu' vivo"
            )

        lock = RecoveryLock(_recovery_lock_path(root, transaction))
        if not lock.acquire():
            errors.append(
                f"transazione {transaction.name}: recovery concorrente gia' in corso"
            )
            continue
        try:
            _recover_one(transaction, manifest, root, recovered, errors)
        finally:
            lock.release()
    return recovered, errors
