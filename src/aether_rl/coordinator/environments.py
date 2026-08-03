from __future__ import annotations

import asyncio
import math
from collections.abc import Iterable
from typing import Literal

import verifiers.v1 as vf
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from verifiers.v1.clients.client import ModelContext
from verifiers.v1.loaders import load_environment
from verifiers.v1.task import task_data_cls
from verifiers.v1.types import SamplingConfig

from aether_rl.protocol import (
    AssignmentLease,
    EnvironmentIdentity,
    FailureEnvelope,
    OpaqueId,
    ResultEnvelope,
    canonical_json_bytes,
    episode_digest,
    policy_manifest_digest,
    result_envelope_bytes,
)

from .inference import InferenceBroker, create_train_client


class EnvironmentSourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: OpaqueId
    kind: Literal["train", "eval"]
    environment: EnvironmentIdentity
    tasks: tuple[dict[str, JsonValue], ...] = Field(min_length=1)
    sampling: SamplingConfig
    group_size: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    result_size_limit_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    assignment_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_canonical_payloads(self) -> EnvironmentSourceSpec:
        if not math.isfinite(self.weight):
            raise ValueError("weight must be finite")
        if self.kind == "train" and (self.sampling.temperature is None or self.sampling.temperature <= 0):
            raise ValueError("train sampling requires a positive temperature")
        canonical_json_bytes(list(self.tasks))
        canonical_json_bytes(self.sampling)
        return self


class EnvironmentCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: tuple[EnvironmentSourceSpec, ...]

    @model_validator(mode="after")
    def validate_source_ids(self) -> EnvironmentCatalog:
        source_ids = tuple(source.source_id for source in self.sources)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source IDs must not contain duplicates")
        return self


def verifier_v1_task_payloads(tasks: Iterable[object]) -> tuple[dict[str, JsonValue], ...]:
    """Convert already-loaded verifier v1 tasks into central JSON payloads."""
    payloads = tuple(task.data.model_dump(mode="json") for task in tasks)  # type: ignore[attr-defined]
    canonical_json_bytes(list(payloads))
    if not payloads:
        raise ValueError("tasks must not be empty")
    return payloads


class CentralEpisodeRunner:
    def __init__(
        self,
        environments: dict[str, vf.EnvConfig],
        broker: InferenceBroker,
        repository,
        database_call,
        *,
        renderer_model_name: str,
        renderer_model_revision: str,
        slots: int,
    ):
        self.environments = environments
        self.broker = broker
        self.repository = repository
        self.database_call = database_call
        self.renderer_model_name = renderer_model_name
        self.renderer_model_revision = renderer_model_revision
        self.gate = asyncio.Semaphore(slots)
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.expires_at: dict[str, float] = {}
        self.watchdog = asyncio.create_task(self._watch_expiry(), name="central-episode-watchdog")

    def start(self, lease: AssignmentLease) -> None:
        if lease.lease_id in self.tasks:
            return
        self.broker.register(lease.lease_id, lease.worker_id, lease.worker_session_id)
        self.expires_at[lease.lease_id] = lease.expires_at
        task = asyncio.create_task(self._execute(lease), name=f"central-episode-{lease.assignment.assignment_id}")
        self.tasks[lease.lease_id] = task
        task.add_done_callback(lambda _task: self.tasks.pop(lease.lease_id, None))

    def renew(self, lease_id: str, expires_at: float) -> None:
        if lease_id in self.tasks:
            self.expires_at[lease_id] = expires_at

    def cancel(self, lease_id: str) -> None:
        task = self.tasks.get(lease_id)
        if task is not None:
            task.cancel()
        self.broker.close(lease_id)

    async def stop(self) -> None:
        for task in self.tasks.values():
            task.cancel()
        await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        self.tasks.clear()
        self.watchdog.cancel()
        await asyncio.gather(self.watchdog, return_exceptions=True)

    async def _watch_expiry(self) -> None:
        while True:
            now = self.repository.clock()
            for lease_id, expires_at in tuple(self.expires_at.items()):
                if expires_at <= now:
                    self.cancel(lease_id)
            await asyncio.sleep(0.5)

    async def _execute(self, lease: AssignmentLease) -> None:
        try:
            async with self.gate:
                envelope = await self._run_environment(lease)
            if len(result_envelope_bytes(envelope)) > lease.assignment.result_size_limit_bytes:
                raise ResultTooLargeError("rollout result exceeds assignment result_size_limit_bytes")
            await self.database_call(self.repository.accept_result, envelope)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            failure = FailureEnvelope(
                assignment_id=lease.assignment.assignment_id,
                attempt=lease.attempt,
                lease_id=lease.lease_id,
                worker_id=lease.worker_id,
                worker_session_id=lease.worker_session_id,
                failed_at=self.repository.clock(),
                code="result_too_large" if isinstance(error, ResultTooLargeError) else "central_execution_failed",
                message=(str(error) or type(error).__name__)[:8192],
                retryable=not isinstance(error, ResultTooLargeError),
            )
            await self.database_call(self.repository.accept_failure, failure)
        finally:
            self.expires_at.pop(lease.lease_id, None)
            self.broker.close(lease.lease_id)

    async def _run_environment(self, lease: AssignmentLease) -> ResultEnvelope:
        assignment = lease.assignment
        env_config = self.environments.get(assignment.source_id)
        if env_config is None:
            raise RuntimeError("assignment requires an unconfigured coordinator environment")
        env = load_environment(env_config)
        served_model_name = assignment.policy.served_model_name
        if served_model_name is None:
            raise RuntimeError("assignment policy does not declare a served model name")
        task_type = type(env.taskset).task_type()
        task_data = task_data_cls(task_type).model_validate(assignment.task_data)
        task = task_type(task_data, env.taskset.config.task)
        client = create_train_client(
            self.broker,
            lease.lease_id,
            renderer_model_name=self.renderer_model_name,
            renderer_model_revision=self.renderer_model_revision,
        )
        try:
            async with env.serving():
                (slot,) = env.slots(task)
                episode = await env.run_slot(
                    slot,
                    ModelContext(model=served_model_name, client=client, sampling=assignment.sampling),
                    asyncio.Semaphore(env.config.max_concurrent) if env.config.max_concurrent else None,
                )
        finally:
            await client.close()
        wire_episode = vf.WireEpisode.model_validate(episode.model_dump(mode="python"))
        digest = policy_manifest_digest(assignment.policy)
        return ResultEnvelope(
            assignment_id=assignment.assignment_id,
            attempt=lease.attempt,
            lease_id=lease.lease_id,
            worker_id=lease.worker_id,
            worker_session_id=lease.worker_session_id,
            requested_policy_id=assignment.policy.policy_id,
            served_policy_id=assignment.policy.policy_id,
            requested_policy_digest=digest,
            served_policy_digest=digest,
            completed_at=self.repository.clock(),
            result_digest=episode_digest(wire_episode),
            episode=wire_episode,
        )


class ResultTooLargeError(RuntimeError):
    pass
