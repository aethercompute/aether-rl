from __future__ import annotations

import asyncio
import importlib.util
import time
from collections.abc import Sequence
from dataclasses import dataclass

import verifiers.v1 as vf
from verifiers.v1.clients.client import ModelContext
from verifiers.v1.clients.config import TrainClientConfig, resolve_client
from verifiers.v1.loaders import load_environment, resolve_env_config
from verifiers.v1.task import task_data_cls
from verifiers.v1.utils.install import env_module, env_name

from aether_rl.configs.worker import WorkerConfig, WorkerEnvironmentConfig
from aether_rl.protocol import (
    AssignmentLease,
    FailureEnvelope,
    ResultEnvelope,
    TerminalEnvelope,
    episode_digest,
    policy_manifest_digest,
    result_envelope_bytes,
)


@dataclass(frozen=True)
class WorkerEnvironment:
    config: WorkerEnvironmentConfig
    env_config: vf.EnvConfig


class TerminalExecutionError(RuntimeError):
    retryable = False

    def __init__(self, message: str, *, code: str = "execution_failed"):
        super().__init__(message)
        self.code = code


class VerifiersAssignmentExecutor:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.environments = {
            environment.id: self._resolve_environment(environment) for environment in config.environments
        }

    async def execute(self, lease: AssignmentLease, cancel_event: asyncio.Event) -> ResultEnvelope:
        assignment = lease.assignment
        environment = self.environments.get(assignment.environment.id)
        if environment is None or environment.config.revision != assignment.environment.revision:
            raise RuntimeError("assignment requires an unconfigured environment")
        if cancel_event.is_set():
            raise asyncio.CancelledError

        env = load_environment(environment.env_config)
        served_model_name = assignment.policy.served_model_name
        if served_model_name is None:
            raise RuntimeError("assignment policy does not declare a served model name")
        task_type = type(env.taskset).task_type()
        task_data = task_data_cls(task_type).model_validate(assignment.task_data)
        task = task_type(task_data, env.taskset.config.task)
        client_config = TrainClientConfig(
            base_url=f"http://127.0.0.1:{self.config.inference_port}/v1",
            renderer_model_name=self.config.base_model.model_name,
            renderer_model_revision=self.config.base_model.tokenizer_revision,
        )
        client = resolve_client(client_config)
        try:
            rollout = asyncio.create_task(self._run_episode(env, task, served_model_name, client, assignment.sampling))
            cancel_waiter = asyncio.create_task(cancel_event.wait())
            try:
                done, _ = await asyncio.wait({rollout, cancel_waiter}, return_when=asyncio.FIRST_COMPLETED)
                if cancel_waiter in done:
                    rollout.cancel()
                    await asyncio.gather(rollout, return_exceptions=True)
                    raise asyncio.CancelledError
                episode = vf.WireEpisode.model_validate((await rollout).model_dump(mode="python"))
            finally:
                if not rollout.done():
                    rollout.cancel()
                    await asyncio.gather(rollout, return_exceptions=True)
                cancel_waiter.cancel()
                await asyncio.gather(cancel_waiter, return_exceptions=True)
        finally:
            await client.close()

        digest = policy_manifest_digest(assignment.policy)
        envelope = ResultEnvelope(
            assignment_id=assignment.assignment_id,
            attempt=lease.attempt,
            lease_id=lease.lease_id,
            worker_id=lease.worker_id,
            worker_session_id=lease.worker_session_id,
            requested_policy_id=assignment.policy.policy_id,
            served_policy_id=assignment.policy.policy_id,
            requested_policy_digest=digest,
            served_policy_digest=digest,
            completed_at=time.time(),
            result_digest=episode_digest(episode),
            episode=episode,
        )
        if len(result_envelope_bytes(envelope)) > assignment.result_size_limit_bytes:
            raise TerminalExecutionError(
                "rollout result exceeds assignment result_size_limit_bytes",
                code="result_too_large",
            )
        return envelope

    async def _run_episode(self, env, task, model: str, client, sampling: vf.SamplingConfig) -> vf.WireEpisode:
        gate = asyncio.Semaphore(env.config.max_concurrent) if env.config.max_concurrent else None
        async with env.serving():
            (slot,) = env.slots(task)
            return await env.run_slot(
                slot,
                ModelContext(
                    model=model,
                    client=client,
                    sampling=sampling,
                ),
                gate,
            )

    async def execute_group(
        self, leases_and_cancellations: Sequence[tuple[AssignmentLease, asyncio.Event]]
    ) -> list[TerminalEnvelope]:
        if not leases_and_cancellations:
            raise ValueError("group execution requires at least one lease")
        leases = [lease for lease, _ in leases_and_cancellations]
        first = leases[0].assignment
        if any(
            (
                lease.assignment.group_id,
                lease.assignment.environment,
                lease.assignment.task_data,
                lease.assignment.sampling,
                lease.assignment.policy,
            )
            != (first.group_id, first.environment, first.task_data, first.sampling, first.policy)
            for lease in leases
        ):
            raise RuntimeError("group execution requires identical assignment inputs")

        environment = self.environments.get(first.environment.id)
        if environment is None or environment.config.revision != first.environment.revision:
            raise RuntimeError("assignment requires an unconfigured environment")
        env = load_environment(environment.env_config)
        served_model_name = first.policy.served_model_name
        if served_model_name is None:
            raise RuntimeError("assignment policy does not declare a served model name")
        task_type = type(env.taskset).task_type()
        task_data = task_data_cls(task_type).model_validate(first.task_data)
        task = task_type(task_data, env.taskset.config.task)
        client_config = TrainClientConfig(
            base_url=f"http://127.0.0.1:{self.config.inference_port}/v1",
            renderer_model_name=self.config.base_model.model_name,
            renderer_model_revision=self.config.base_model.tokenizer_revision,
        )
        client = resolve_client(client_config)
        try:
            gate = asyncio.Semaphore(env.config.max_concurrent) if env.config.max_concurrent else None
            async with env.serving():
                slots = env.slots(task, n=len(leases))
                results = await asyncio.gather(
                    *(
                        self._run_group_slot(env, slot, lease, cancel_event, served_model_name, client, gate)
                        for slot, (lease, cancel_event) in zip(slots, leases_and_cancellations, strict=True)
                    ),
                    return_exceptions=True,
                )
        finally:
            await client.close()

        envelopes: list[TerminalEnvelope] = []
        for lease, result in zip(leases, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                raise result
            if isinstance(result, BaseException):
                envelopes.append(self._failure_envelope(lease, result))
            else:
                envelopes.append(result)
        return envelopes

    async def _run_group_slot(self, env, slot, lease, cancel_event, model, client, gate) -> ResultEnvelope:
        rollout = asyncio.create_task(
            env.run_slot(
                slot,
                ModelContext(
                    model=model,
                    client=client,
                    sampling=lease.assignment.sampling,
                ),
                gate,
            )
        )
        cancel_waiter = asyncio.create_task(cancel_event.wait())
        try:
            done, _ = await asyncio.wait({rollout, cancel_waiter}, return_when=asyncio.FIRST_COMPLETED)
            if cancel_waiter in done:
                rollout.cancel()
                await asyncio.gather(rollout, return_exceptions=True)
                raise asyncio.CancelledError
            episode = vf.WireEpisode.model_validate((await rollout).model_dump(mode="python"))
        finally:
            if not rollout.done():
                rollout.cancel()
                await asyncio.gather(rollout, return_exceptions=True)
            cancel_waiter.cancel()
            await asyncio.gather(cancel_waiter, return_exceptions=True)
        envelope = self._result_envelope(lease, episode)
        if len(result_envelope_bytes(envelope)) > lease.assignment.result_size_limit_bytes:
            raise TerminalExecutionError(
                "rollout result exceeds assignment result_size_limit_bytes",
                code="result_too_large",
            )
        return envelope

    @staticmethod
    def _result_envelope(lease: AssignmentLease, episode: vf.WireEpisode) -> ResultEnvelope:
        digest = policy_manifest_digest(lease.assignment.policy)
        return ResultEnvelope(
            assignment_id=lease.assignment.assignment_id,
            attempt=lease.attempt,
            lease_id=lease.lease_id,
            worker_id=lease.worker_id,
            worker_session_id=lease.worker_session_id,
            requested_policy_id=lease.assignment.policy.policy_id,
            served_policy_id=lease.assignment.policy.policy_id,
            requested_policy_digest=digest,
            served_policy_digest=digest,
            completed_at=time.time(),
            result_digest=episode_digest(episode),
            episode=episode,
        )

    @staticmethod
    def _failure_envelope(lease: AssignmentLease, error: BaseException) -> FailureEnvelope:
        return FailureEnvelope(
            assignment_id=lease.assignment.assignment_id,
            attempt=lease.attempt,
            lease_id=lease.lease_id,
            worker_id=lease.worker_id,
            worker_session_id=lease.worker_session_id,
            failed_at=time.time(),
            code=getattr(error, "code", "execution_failed"),
            message=(str(error) or type(error).__name__)[:8192],
            retryable=getattr(error, "retryable", True),
        )

    @staticmethod
    def _resolve_environment(config: WorkerEnvironmentConfig) -> WorkerEnvironment:
        env_config = resolve_env_config(config.config)
        if env_config.env_id != config.id:
            raise RuntimeError(f"environment config resolves to {env_config.env_id!r}, expected {config.id!r}")
        package_ids = _environment_package_ids(env_config)
        if config.package not in package_ids:
            raise RuntimeError("environment package must match the configured verifiers taskset or environment id")
        return WorkerEnvironment(config=config, env_config=env_config)


def _environment_package_ids(env_config: vf.EnvConfig) -> set[str]:
    package_ids: set[str] = set()
    for plugin_id in (env_config.taskset.id, env_config.id):
        if not plugin_id:
            continue
        package_ids.update({plugin_id, env_name(plugin_id), env_module(plugin_id)})
        for namespace in ("tasksets", "envs"):
            if importlib.util.find_spec(f"verifiers.v1.{namespace}.{env_module(plugin_id)}") is not None:
                package_ids.add("verifiers")
    return package_ids
