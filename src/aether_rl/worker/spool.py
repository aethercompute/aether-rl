from __future__ import annotations

import fcntl
import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from aether_rl.protocol import (
    FailureEnvelope,
    ResultEnvelope,
    SubmissionResponse,
    TerminalEnvelope,
    canonical_json_bytes,
    decode_result_envelope,
    result_envelope_bytes,
    sha256_digest,
)


class WorkerStateError(RuntimeError):
    pass


class SpoolCorruptionError(WorkerStateError):
    pass


@dataclass(frozen=True)
class SpoolEntry:
    path: Path
    kind: Literal["result", "failure"]
    digest: str
    body: bytes
    envelope: TerminalEnvelope

    @property
    def content_type(self) -> str:
        return "application/msgpack" if self.kind == "result" else "application/json"


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

    def __enter__(self) -> "WorkerState":
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


class WorkerSpool:
    def __init__(self, state: WorkerState):
        self.state = state
        self.root = state.root / "spool"
        self.incoming = self.root / "incoming"
        self.pending = self.root / "pending"
        self.rejected = self.root / "rejected"
        for directory in (self.root, self.incoming, self.pending, self.rejected):
            state._create_directory(directory)
        for path in self.incoming.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()
        state._fsync_directory(self.incoming)
        self._entries: dict[Path, SpoolEntry] = {}
        for path in sorted(self.pending.iterdir(), key=lambda item: (item.lstat().st_mtime_ns, item.name)):
            try:
                self._entries[path] = self._load(path)
            except Exception as error:
                self._quarantine(path)
                raise SpoolCorruptionError(f"invalid pending worker spool entry: {path.name}") from error

    def publish(self, envelope: TerminalEnvelope) -> SpoolEntry:
        if isinstance(envelope, ResultEnvelope):
            kind: Literal["result", "failure"] = "result"
            body = result_envelope_bytes(envelope)
            suffix = ".result.msgpack"
        else:
            kind = "failure"
            body = canonical_json_bytes(envelope)
            suffix = ".failure.json"
        digest = sha256_digest(body)
        final_path = self.pending / f"{digest.removeprefix('sha256:')}{suffix}"
        temporary = self.incoming / f"{uuid.uuid4().hex}.tmp"
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(body)
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temporary, final_path)
            except FileExistsError:
                existing = self._load(final_path)
                if existing.digest != digest or existing.body != body:
                    raise SpoolCorruptionError("existing spool entry conflicts with terminal envelope")
                self._entries[final_path] = existing
                return existing
            self.state._fsync_directory(self.pending)
        finally:
            temporary.unlink(missing_ok=True)
            self.state._fsync_directory(self.incoming)
        entry = SpoolEntry(final_path, kind, digest, body, envelope)
        self._entries[final_path] = entry
        return entry

    def entries(self) -> tuple[SpoolEntry, ...]:
        paths = set(self.pending.iterdir())
        for path in sorted(paths - self._entries.keys(), key=lambda item: (item.lstat().st_mtime_ns, item.name)):
            try:
                self._entries[path] = self._load(path)
            except Exception as error:
                self._quarantine(path)
                raise SpoolCorruptionError(f"invalid pending worker spool entry: {path.name}") from error
        for path in self._entries.keys() - paths:
            self._entries.pop(path)
        return tuple(self._entries.values())

    def acknowledge(self, entry: SpoolEntry, response: SubmissionResponse) -> None:
        if response.assignment_id != entry.envelope.assignment_id or response.envelope_digest != entry.digest:
            raise SpoolCorruptionError("submission acknowledgement does not match the spooled envelope")
        entry.path.unlink()
        self._entries.pop(entry.path, None)
        self.state._fsync_directory(self.pending)

    def reject(self, entry: SpoolEntry) -> Path:
        destination = self.rejected / entry.path.name
        try:
            os.link(entry.path, destination)
        except FileExistsError:
            if destination.read_bytes() != entry.body:
                raise SpoolCorruptionError("rejected spool entry conflicts with pending bytes")
        self.state._fsync_directory(self.rejected)
        entry.path.unlink()
        self._entries.pop(entry.path, None)
        self.state._fsync_directory(self.pending)
        return destination

    def _load(self, path: Path) -> SpoolEntry:
        if path.is_symlink():
            raise SpoolCorruptionError("spool entries must not be symlinks")
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise SpoolCorruptionError("spool entry is not a regular file")
        match = re.fullmatch(r"([0-9a-f]{64})\.(result\.msgpack|failure\.json)", path.name)
        if match is None:
            raise SpoolCorruptionError("spool filename is malformed")
        body = path.read_bytes()
        digest = sha256_digest(body)
        if digest != f"sha256:{match.group(1)}":
            raise SpoolCorruptionError("spool filename digest does not match its bytes")
        if match.group(2) == "result.msgpack":
            kind: Literal["result", "failure"] = "result"
            envelope: TerminalEnvelope = decode_result_envelope(body)
            canonical = result_envelope_bytes(envelope)
        else:
            kind = "failure"
            envelope = FailureEnvelope.model_validate_json(body)
            canonical = canonical_json_bytes(envelope)
        if canonical != body:
            raise SpoolCorruptionError("spool entry is not canonically encoded")
        return SpoolEntry(path, kind, digest, body, envelope)

    def _quarantine(self, path: Path) -> None:
        destination = self.rejected / path.name
        if destination.exists() or destination.is_symlink():
            raise SpoolCorruptionError("cannot quarantine over an existing rejected entry")
        os.replace(path, destination)
        self.state._fsync_directory(self.rejected)
        self.state._fsync_directory(self.pending)
