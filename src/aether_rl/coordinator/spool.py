from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from pathlib import Path


class ImmutableArtifactConflictError(ValueError):
    pass


class AtomicSpool:
    def __init__(self, run_root: Path):
        self.run_root = run_root.resolve()
        self.spool_dir = self.run_root / "spool"
        self.incoming_dir = self.spool_dir / "incoming"
        self.results_dir = self.spool_dir / "results"
        for directory in (self.spool_dir, self.incoming_dir, self.results_dir):
            self._create_durable_directory(directory)

    def publish_result(self, digest: str, data: bytes, *, suffix: str = ".msgpack") -> str:
        self._validate_directory(self.incoming_dir)
        self._validate_directory(self.results_dir)
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError("result artifact digest must be a SHA-256 digest")
        if suffix not in {".json", ".msgpack"}:
            raise ValueError("result artifact suffix is unsupported")
        digest_hex = digest.removeprefix("sha256:")
        final_path = self.results_dir / f"{digest_hex}{suffix}"
        temporary_path = self.incoming_dir / f"{uuid.uuid4().hex}.tmp"
        descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temporary_path, final_path)
            except FileExistsError:
                if (
                    final_path.is_symlink()
                    or final_path.stat().st_size != len(data)
                    or self.file_digest(final_path) != digest
                ):
                    raise ImmutableArtifactConflictError(f"immutable result artifact conflicts: {final_path}")
            self._fsync_directory(self.results_dir)
        finally:
            temporary_path.unlink(missing_ok=True)
        return final_path.relative_to(self.run_root).as_posix()

    def resolve_result(self, relative_path: str) -> Path:
        self._validate_directory(self.results_dir)
        raw_path = self.run_root / relative_path
        if raw_path.is_symlink():
            raise ValueError("result artifact path must not be a symlink")
        path = raw_path.resolve()
        try:
            path.relative_to(self.results_dir.resolve())
        except ValueError as error:
            raise ValueError("result artifact path escapes the result spool") from error
        if path.parent != self.results_dir.resolve():
            raise ValueError("result artifact path is not a regular spool entry")
        return path

    @staticmethod
    def file_digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"sha256:{digest.hexdigest()}"

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @classmethod
    def _create_durable_directory(cls, path: Path) -> None:
        if path.is_symlink():
            raise ValueError(f"spool directory must not be a symlink: {path}")
        try:
            mode = path.stat().st_mode
        except FileNotFoundError:
            path.mkdir()
            cls._fsync_directory(path.parent)
            mode = path.stat().st_mode
        if not stat.S_ISDIR(mode):
            raise ValueError(f"spool path must be a directory: {path}")

    @staticmethod
    def _validate_directory(path: Path) -> None:
        if path.is_symlink() or not stat.S_ISDIR(path.stat().st_mode):
            raise ValueError(f"spool path must be a non-symlink directory: {path}")
