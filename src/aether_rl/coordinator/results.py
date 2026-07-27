from __future__ import annotations

import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import msgspec
import verifiers.v1 as vf

from aether_rl.orchestrator.envs import episode_to_rollouts
from aether_rl.orchestrator.filters import RolloutFilter, apply_filters
from aether_rl.orchestrator.trajectories import trace_to_samples
from aether_rl.orchestrator.types import Rollout
from aether_rl.protocol import (
    EnvironmentIdentity,
    ResultEnvelope,
    canonical_json_bytes,
    decode_result_envelope,
    policy_manifest_digest,
    sha256_digest,
)
from aether_rl.transport import TrainingBatch, TrainingSample

from .database import (
    ArtifactCorruptionError,
    ConflictError,
    CoordinatorRepository,
    GroupOutcome,
    ReadyGroup,
    TrainingBatchRecord,
)

if TYPE_CHECKING:
    from aether_rl.orchestrator.algo import Algorithm


@dataclass(frozen=True)
class ResultProcessingSource:
    source_id: str
    environment: EnvironmentIdentity
    processing_id: str
    algorithm: Algorithm | None = None
    pre_filters: tuple[RolloutFilter, ...] = ()
    post_filters: tuple[RolloutFilter, ...] = ()
    mm_token_type_ids_mapping: dict[int, int] | None = None
    requires_group_scoring: bool = False


class ProcessedRolloutPayload(msgspec.Struct):
    samples: list[TrainingSample]
    token_count: int


class ProcessedGroupPayload(msgspec.Struct):
    group_id: str
    source_id: str
    kind: str
    policy_id: str
    policy_version: int
    input_digest: str
    rollouts: list[ProcessedRolloutPayload]
    evaluation_records: list[dict[str, Any]]


class DurableTrainingQueue:
    def __init__(self, run_root: Path):
        self.run_root = run_root.resolve()
        self.root = self.run_root / "training-queue"
        self.incoming = self.root / "incoming"
        self.groups = self.root / "groups"
        self.batches = self.root / "batches"
        for directory in (self.root, self.incoming, self.groups, self.batches):
            self._create_directory(directory)

    def publish_group(self, digest: str, data: bytes) -> Path:
        self._validate_digest(digest)
        return self._publish(self.groups / f"{digest.removeprefix('sha256:')}.msgpack", digest, data)

    def publish_batch(self, step: int, digest: str, data: bytes) -> Path:
        if step < 1:
            raise ValueError("training batch step must be positive")
        self._validate_digest(digest)
        step_dir = self.batches / f"step_{step}"
        self._create_directory(step_dir)
        return self._publish(step_dir / "train_rollouts.bin", digest, data)

    def verify(self, path: Path, digest: str, size_bytes: int) -> bytes:
        resolved = path.resolve(strict=True)
        if path.is_symlink() or resolved.parent not in {self.groups.resolve(), *self._batch_directories()}:
            raise ArtifactCorruptionError("training queue artifact path is unsafe")
        metadata = resolved.stat()
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != size_bytes:
            raise ArtifactCorruptionError("training queue artifact size does not match durable state")
        data = resolved.read_bytes()
        if sha256_digest(data) != digest:
            raise ArtifactCorruptionError("training queue artifact digest does not match durable state")
        return data

    def recover(self, repository: CoordinatorRepository) -> None:
        for path in self.incoming.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()
        referenced: set[Path] = set()
        for record in repository.processed_groups():
            self.verify(record.artifact_path, record.artifact_digest, record.size_bytes)
            referenced.add(record.artifact_path.resolve())
        for record in repository.training_batches():
            self.verify(record.artifact_path, record.artifact_digest, record.size_bytes)
            referenced.add(record.artifact_path.resolve())
        for path in self.groups.iterdir():
            if (path.is_file() or path.is_symlink()) and path.resolve() not in referenced:
                path.unlink()
        for step_dir in self.batches.iterdir():
            if step_dir.is_symlink() or not step_dir.is_dir():
                raise ArtifactCorruptionError("training batch directory is unsafe")
            for path in step_dir.iterdir():
                if (path.is_file() or path.is_symlink()) and path.resolve() not in referenced:
                    path.unlink()
            if not any(step_dir.iterdir()):
                step_dir.rmdir()

    def _publish(self, final_path: Path, digest: str, data: bytes) -> Path:
        self._validate_directory(self.incoming)
        self._validate_directory(final_path.parent)
        if sha256_digest(data) != digest:
            raise ValueError("training queue data does not match its digest")
        temporary_path = self.incoming / f"{uuid.uuid4().hex}.tmp"
        descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(descriptor, "wb") as file:
                file.write(data)
                file.flush()
                os.fsync(file.fileno())
            try:
                os.link(temporary_path, final_path)
            except FileExistsError:
                self.verify(final_path, digest, len(data))
            self._fsync_directory(final_path.parent)
        finally:
            temporary_path.unlink(missing_ok=True)
        return final_path

    def _batch_directories(self) -> set[Path]:
        return {path.resolve() for path in self.batches.iterdir() if path.is_dir() and not path.is_symlink()}

    @staticmethod
    def _validate_digest(digest: str) -> None:
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ValueError("artifact digest must be a SHA-256 digest")

    @classmethod
    def _create_directory(cls, path: Path) -> None:
        if path.exists() or path.is_symlink():
            cls._validate_directory(path)
            return
        path.mkdir(mode=0o700)
        cls._fsync_directory(path.parent)

    @staticmethod
    def _validate_directory(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise ArtifactCorruptionError(f"training queue directory is unsafe: {path}")

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


class RemoteResultProcessor:
    def __init__(
        self,
        repository: CoordinatorRepository,
        sources: tuple[ResultProcessingSource, ...],
        *,
        batch_size: int | None = None,
        token_batch_size: int | None = None,
    ):
        if (batch_size is None) == (token_batch_size is None):
            raise ValueError("exactly one batch size must be configured")
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch size must be positive")
        if token_batch_size is not None and token_batch_size < 1:
            raise ValueError("token batch size must be positive")
        self.repository = repository
        self.sources = {source.source_id: source for source in sources}
        if len(self.sources) != len(sources):
            raise ValueError("result processing source IDs must be unique")
        self.batch_size = batch_size
        self.token_batch_size = token_batch_size
        self.queue = DurableTrainingQueue(repository.run_root)
        self.queue.recover(repository)
        self._encoder = msgspec.msgpack.Encoder()
        self._group_decoder = msgspec.msgpack.Decoder(type=ProcessedGroupPayload)

    async def process_available(self) -> tuple[int, int]:
        processed_groups = 0
        while groups := self.repository.ready_groups():
            await self._process_group(groups[0])
            processed_groups += 1
        emitted_batches = self._emit_ready_batches()
        return processed_groups, emitted_batches

    async def _process_group(self, group: ReadyGroup) -> None:
        if group.source_id is None or group.source_id not in self.sources:
            raise ConflictError(f"group {group.group_id} has no configured processing source")
        source = self.sources[group.source_id]
        if any(outcome.assignment.environment != source.environment for outcome in group.outcomes):
            raise ConflictError("group environment does not match its processing source")
        input_digest = self._input_digest(group, source)
        episodes = [self._load_episode(outcome) for outcome in group.outcomes]
        rollouts: list[Rollout] = []
        for outcome, episode in zip(group.outcomes, episodes, strict=True):
            try:
                episode_rollouts = episode_to_rollouts(episode)
            except RuntimeError as error:
                episode_rollouts = [self._failure_rollout(outcome, error)]
            for rollout in episode_rollouts:
                self._stamp_rollout(rollout, group, outcome)
                if not rollout.has_error and rollout.num_turns == 0:
                    rollout.errors.append(vf.Error(type="EmptyTrajectory", message="Rollout returned no turns"))
                    rollout.ok = False
            rollouts.extend(episode_rollouts)

        evaluation_records: list[dict[str, Any]] = []
        processed_rollouts: list[ProcessedRolloutPayload] = []
        if group.kind == "eval":
            evaluation_records = [rollout.to_record() for rollout in rollouts]
        else:
            if source.algorithm is None:
                raise ConflictError("train processing source has no algorithm")
            for rollout in rollouts:
                if rollout.has_error or not rollout.trainable:
                    continue
                rollout.samples = trace_to_samples(
                    rollout,
                    env_name=rollout.env_name,
                    mm_token_type_ids_mapping=source.mm_token_type_ids_mapping,
                )
                for sample in rollout.samples:
                    sample.behavior_policy_id = rollout.policy_id
                    sample.behavior_policy_version = rollout.policy_version
                    sample.behavior_policy_digest = rollout.policy_digest
                await source.algorithm.finalize_rollout(rollout)
            has_error = any(rollout.has_error for rollout in rollouts)
            survivors = [rollout for rollout in rollouts if not rollout.has_error and rollout.trainable]
            if source.requires_group_scoring and has_error:
                survivors = []
            if survivors:
                await source.algorithm.finalize_group(survivors)
                temperature = group.outcomes[0].assignment.sampling.temperature
                if temperature is None:
                    raise ConflictError("train assignment sampling temperature is missing")
                for rollout in survivors:
                    for sample in rollout.samples:
                        sample.temperatures = [temperature] * len(sample.token_ids)
                apply_filters(list(source.pre_filters), survivors)
                survivors = [rollout for rollout in survivors if not rollout.is_filtered]
                for rollout in survivors:
                    rollout.filter_results = {}
                    rollout.is_filtered = False
                apply_filters(list(source.post_filters), survivors)
                processed_rollouts = [
                    ProcessedRolloutPayload(
                        samples=[] if rollout.is_filtered else rollout.samples,
                        token_count=sum(len(sample.token_ids) for sample in rollout.samples),
                    )
                    for rollout in survivors
                    if rollout.samples
                ]

        payload = ProcessedGroupPayload(
            group_id=group.group_id,
            source_id=source.source_id,
            kind=group.kind,
            policy_id=group.outcomes[0].assignment.policy.policy_id,
            policy_version=group.outcomes[0].assignment.policy.policy_version,
            input_digest=input_digest,
            rollouts=processed_rollouts,
            evaluation_records=evaluation_records,
        )
        artifact_bytes = self._encoder.encode(payload)
        artifact_digest = sha256_digest(artifact_bytes)
        artifact_path = self.queue.publish_group(artifact_digest, artifact_bytes)
        token_counts = [rollout.token_count for rollout in processed_rollouts]
        self.repository.record_processed_group(
            group.group_id,
            input_digest=input_digest,
            artifact_digest=artifact_digest,
            artifact_path=artifact_path,
            size_bytes=len(artifact_bytes),
            token_counts=token_counts,
        )

    def _emit_ready_batches(self) -> int:
        emitted = 0
        while True:
            pending = self.repository.pending_processed_rollouts()
            selected = []
            tokens = 0
            for rollout in pending:
                selected.append(rollout)
                tokens += rollout.token_count
                if self.batch_size is not None and len(selected) >= self.batch_size:
                    break
                if self.token_batch_size is not None and tokens >= self.token_batch_size:
                    break
            ready = (
                len(selected) >= self.batch_size
                if self.batch_size is not None
                else tokens >= (self.token_batch_size or 0)
            )
            if not ready:
                return emitted
            payloads: dict[Path, ProcessedGroupPayload] = {}
            samples: list[TrainingSample] = []
            members: list[tuple[str, int]] = []
            for rollout in selected:
                payload = payloads.get(rollout.artifact_path)
                if payload is None:
                    artifact = self.queue.verify(
                        rollout.artifact_path,
                        rollout.artifact_digest,
                        rollout.size_bytes,
                    )
                    payload = self._group_decoder.decode(artifact)
                    payloads[rollout.artifact_path] = payload
                samples.extend(payload.rollouts[rollout.ordinal].samples)
                members.append((rollout.group_id, rollout.ordinal))
            if not samples:
                self.repository.discard_processed_rollouts(members)
                continue
            step = self.repository.next_training_batch_step()
            batch_bytes = self._encoder.encode(TrainingBatch(examples=samples, step=step))
            batch_digest = sha256_digest(batch_bytes)
            batch_path = self.queue.publish_batch(step, batch_digest, batch_bytes)
            self.repository.record_training_batch(
                step=step,
                artifact_digest=batch_digest,
                artifact_path=batch_path,
                size_bytes=len(batch_bytes),
                sample_count=len(samples),
                members=members,
            )
            emitted += 1

    def _load_episode(self, outcome: GroupOutcome) -> vf.WireEpisode:
        if outcome.outcome != "result":
            return vf.WireEpisode(
                id=outcome.assignment.assignment_id,
                env=outcome.assignment.environment.id,
                ok=False,
                errors=[vf.Error(type=outcome.outcome, message="Assignment ended without a rollout result")],
            )
        if outcome.result_path is None or outcome.envelope_digest is None:
            raise ArtifactCorruptionError("result outcome is missing its durable artifact")
        data = outcome.result_path.read_bytes()
        if sha256_digest(data) != outcome.envelope_digest:
            raise ArtifactCorruptionError("result artifact digest changed before processing")
        envelope = (
            ResultEnvelope.model_validate_json(data)
            if outcome.result_path.suffix == ".json"
            else decode_result_envelope(data)
        )
        if envelope.assignment_id != outcome.assignment.assignment_id:
            raise ArtifactCorruptionError("result artifact belongs to a different assignment")
        if envelope.episode.env != outcome.assignment.environment.id:
            raise ConflictError("result episode environment does not match its assignment")
        return envelope.episode

    @staticmethod
    def _failure_rollout(outcome: GroupOutcome, error: Exception) -> Rollout:
        task_data = vf.WireTaskData.model_validate(outcome.assignment.task_data)
        rollout = Rollout(task=vf.TraceTask(type="Task", data=task_data))
        rollout.capture_error(error)
        return rollout

    @staticmethod
    def _stamp_rollout(rollout: Rollout, group: ReadyGroup, outcome: GroupOutcome) -> None:
        trace_task = rollout.task.data.model_dump(mode="json")
        if any(trace_task.get(key) != value for key, value in outcome.assignment.task_data.items()):
            raise ConflictError("result trace task does not match its assignment")
        rollout.kind = group.kind
        rollout.env_name = outcome.assignment.environment.id
        rollout.group_id = group.group_id
        rollout.policy_version = outcome.assignment.policy.policy_version
        rollout.policy_id = outcome.assignment.policy.policy_id
        rollout.policy_digest = policy_manifest_digest(outcome.assignment.policy)
        expected_model = outcome.assignment.policy.served_model_name
        if any(call.model is not None and call.model != expected_model for call in rollout.calls):
            raise ConflictError("result trace model does not match its assigned policy")

    @staticmethod
    def _input_digest(group: ReadyGroup, source: ResultProcessingSource) -> str:
        return sha256_digest(
            canonical_json_bytes(
                {
                    "group_id": group.group_id,
                    "source_id": source.source_id,
                    "processing_id": source.processing_id,
                    "outcomes": [
                        {
                            "assignment_id": outcome.assignment.assignment_id,
                            "outcome": outcome.outcome,
                            "envelope_digest": outcome.envelope_digest,
                        }
                        for outcome in group.outcomes
                    ],
                }
            )
        )


def decode_training_batch(record: TrainingBatchRecord) -> TrainingBatch:
    data = record.artifact_path.read_bytes()
    if len(data) != record.size_bytes or sha256_digest(data) != record.artifact_digest:
        raise ArtifactCorruptionError("training batch artifact does not match durable state")
    return msgspec.msgpack.decode(data, type=TrainingBatch)
