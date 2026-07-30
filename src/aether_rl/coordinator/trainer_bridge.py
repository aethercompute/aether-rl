from __future__ import annotations

import hashlib
import os
import uuid
from pathlib import Path

from aether_rl.coordinator.database import ArtifactCorruptionError, CoordinatorRepository, TrainingBatchRecord
from aether_rl.protocol import sha256_digest
from aether_rl.transport.filesystem import BATCH_FILE_NAME, BATCH_FILE_TMP_NAME
from aether_rl.utils.pathing import get_rollout_dir, get_step_path


class CoordinatorTrainingBatchExporter:
    def __init__(self, repository: CoordinatorRepository, trainer_output_dir: Path, *, run_id: str, run_config: bytes):
        if not run_id.startswith("run_"):
            raise ValueError("trainer run_id must start with 'run_' for existing trainer discovery")
        if not run_config:
            raise ValueError("trainer run configuration must not be empty")
        self.repository = repository
        self.trainer_output_dir = Path(trainer_output_dir)
        self.run_id = run_id
        self.run_config = run_config
        self.last_exported_step = 0

    def export_available(self, *, limit: int | None = None) -> int:
        if limit is not None and limit < 1:
            raise ValueError("export limit must be positive")
        self._export_run_config()
        exported = 0
        for record in self.repository.training_batches(after_step=self.last_exported_step):
            if limit is not None and exported >= limit:
                break
            if self._export_record(record):
                exported += 1
            self.last_exported_step = record.step
        return exported

    def _export_record(self, record: TrainingBatchRecord) -> bool:
        final_path = self._batch_path(record.step)
        if final_path.is_symlink():
            raise ArtifactCorruptionError("exported trainer batch path is unsafe")
        if final_path.is_file():
            if (
                final_path.stat().st_size != record.size_bytes
                or self._file_digest(final_path) != record.artifact_digest
            ):
                raise ArtifactCorruptionError("exported trainer batch conflicts with coordinator state")
            return False
        data = record.artifact_path.read_bytes()
        if len(data) != record.size_bytes or sha256_digest(data) != record.artifact_digest:
            raise ArtifactCorruptionError("training batch artifact does not match durable state")
        return self._publish_file(
            final_path, data, conflict_message="exported trainer batch conflicts with coordinator state"
        )

    @staticmethod
    def _file_digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with open(path, "rb") as file:
            while chunk := file.read(1024 * 1024):
                hasher.update(chunk)
        return f"sha256:{hasher.hexdigest()}"

    def _export_run_config(self) -> None:
        final_path = self.trainer_output_dir / self.run_id / "control" / "orch.toml"
        self._ensure_directory(final_path.parent)
        if final_path.is_symlink():
            raise ArtifactCorruptionError("trainer run config path is unsafe")
        if final_path.is_file() and final_path.read_bytes() == self.run_config:
            return
        temporary_path = final_path.parent / f".{BATCH_FILE_TMP_NAME}.{uuid.uuid4().hex}"
        descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(self.run_config)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary_path, final_path)
            self._fsync_directory(final_path.parent)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _publish_file(self, final_path: Path, data: bytes, *, conflict_message: str) -> bool:
        self._ensure_directory(final_path.parent)
        temporary_path = final_path.parent / f".{BATCH_FILE_TMP_NAME}.{uuid.uuid4().hex}"
        descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temporary_path, final_path)
            except FileExistsError:
                if final_path.is_symlink() or final_path.read_bytes() != data:
                    raise ArtifactCorruptionError(conflict_message) from None
                return False
            self._fsync_directory(final_path.parent)
        finally:
            temporary_path.unlink(missing_ok=True)
        return True

    def _batch_path(self, step: int) -> Path:
        run_dir = self.trainer_output_dir / self.run_id
        return get_step_path(get_rollout_dir(run_dir), step) / BATCH_FILE_NAME

    @classmethod
    def _ensure_directory(cls, path: Path) -> None:
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not path.is_dir():
                raise ArtifactCorruptionError(f"trainer export directory is unsafe: {path}")
            return
        cls._ensure_directory(path.parent)
        path.mkdir(mode=0o700)
        cls._fsync_directory(path.parent)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
