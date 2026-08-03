from __future__ import annotations

import fcntl
import os
import re
import stat
import uuid
from pathlib import Path
from typing import BinaryIO


class WorkerStateError(RuntimeError):
    pass


class WorkerState:
    def __init__(self, root: Path):
        if root.is_symlink():
            raise WorkerStateError("worker state directory must not be a symlink")
        self.root = root.absolute()
        self._create_directory(self.root)
        lock_path = self.root / "worker.lock"
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        os.fchmod(descriptor, 0o600)
        self._lock_file: BinaryIO = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._lock_file.close()
            raise WorkerStateError("worker state directory is already owned by another process") from error
        self.identity_dir = self.root / "identity"
        self._create_directory(self.identity_dir)

    def load_or_create_worker_id(self) -> str:
        path = self.identity_dir / "worker-id"
        if path.exists() or path.is_symlink():
            try:
                descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
            except OSError as error:
                raise WorkerStateError("worker identity path is unsafe") from error
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
                    raise WorkerStateError("worker identity file is unsafe")
                if stat.S_IMODE(metadata.st_mode) != 0o600:
                    os.fchmod(descriptor, 0o600)
                    os.fsync(descriptor)
                with os.fdopen(os.dup(descriptor), "r") as file:
                    worker_id = file.read().strip()
            finally:
                os.close(descriptor)
            if re.fullmatch(r"worker-[0-9a-f]{32}", worker_id) is None:
                raise WorkerStateError("persisted worker identity is malformed")
            return worker_id
        worker_id = f"worker-{uuid.uuid4().hex}"
        temporary = self.identity_dir / f".{uuid.uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "w") as file:
                file.write(worker_id + "\n")
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                return self.load_or_create_worker_id()
            self._fsync_directory(self.identity_dir)
        finally:
            temporary.unlink(missing_ok=True)
        return worker_id

    def close(self) -> None:
        if not self._lock_file.closed:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()

    def __enter__(self) -> WorkerState:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @classmethod
    def _create_directory(cls, path: Path) -> None:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_dir():
                raise WorkerStateError(f"worker state directory is unsafe: {path}")
            metadata = path.stat()
            if metadata.st_uid != os.getuid():
                raise WorkerStateError(f"worker state directory is owned by another user: {path}")
            if stat.S_IMODE(metadata.st_mode) != 0o700:
                path.chmod(0o700)
                cls._fsync_directory(path.parent)
            return
        if not path.parent.exists():
            cls._create_directory(path.parent)
        elif path.parent.is_symlink() or not path.parent.is_dir():
            raise WorkerStateError(f"worker state parent directory is unsafe: {path.parent}")
        path.mkdir(mode=0o700)
        cls._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
