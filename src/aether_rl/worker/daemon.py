from __future__ import annotations

import asyncio
import importlib.metadata
import os
import platform
import time
import uuid
from collections.abc import Callable, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Protocol

import httpx
import torch

from aether_rl.configs.worker import WorkerConfig
from aether_rl.protocol import (
    AssignmentLease,
    BaseModelIdentity,
    EnvironmentIdentity,
    FailureEnvelope,
    LeaseRequest,
    RuntimeIdentity,
    TerminalEnvelope,
    WorkerCapabilities,
    WorkerHeartbeat,
    WorkerRegistration,
    policy_manifest_digest,
)

from .client import CoordinatorAPIError, CoordinatorClient, CoordinatorProtocolError
from .spool import WorkerSpool, WorkerState


class AssignmentExecutor(Protocol):
    async def execute(self, lease: AssignmentLease, cancel_event: asyncio.Event) -> TerminalEnvelope: ...


class PolicyRuntime(Protocol):
    async def start(self, stop_event: asyncio.Event) -> None: ...
    async def stop(self) -> None: ...
    async def monitor(self) -> None: ...
    def loaded_policy_ids(self) -> tuple[str, ...]: ...
    def acquire(self, manifest) -> AbstractAsyncContextManager[str]: ...


@dataclass
class ActiveAssignment:
    lease: AssignmentLease
    cancel_event: asyncio.Event
    expires_at: float


@dataclass
class InferenceMetrics:
    rollouts_completed: int = 0
    output_tokens: int = 0
    queue_wait_seconds: float = 0.0
    generation_seconds: float = 0.0
    execution_seconds: float = 0.0
    grouped_batches: int = 0
    grouped_batch_members: int = 0
    started_at: float = time.monotonic()

    def record(self, lease: AssignmentLease, envelope: TerminalEnvelope, execution_seconds: float) -> None:
        self.rollouts_completed += 1
        self.output_tokens += _episode_output_tokens(envelope)
        self.queue_wait_seconds += max(0.0, lease.issued_at - lease.assignment.created_at)
        self.generation_seconds += _episode_generation_seconds(envelope)
        self.execution_seconds += max(0.0, execution_seconds)

    def record_batch(self, size: int) -> None:
        if size > 1:
            self.grouped_batches += 1
            self.grouped_batch_members += size

    def snapshot(self, *, running_requests: int = 0, waiting_requests: int = 0) -> dict[str, float]:
        elapsed_hours = max((time.monotonic() - self.started_at) / 3600, 1e-12)
        elapsed_seconds = max(time.monotonic() - self.started_at, 1e-12)
        completed = max(self.rollouts_completed, 1)
        return {
            "inference/agg/rollouts_completed": float(self.rollouts_completed),
            "inference/agg/throughput": self.output_tokens / elapsed_seconds,
            "inference/agg/rollouts_per_hour": self.rollouts_completed / elapsed_hours,
            "inference/agg/queue_wait": self.queue_wait_seconds / completed,
            "inference/agg/generation_time": self.generation_seconds / completed,
            "inference/agg/execution_time": self.execution_seconds / completed,
            "inference/agg/batch_size": self.grouped_batch_members / max(self.grouped_batches, 1),
            "inference/agg/running_requests": float(running_requests),
            "inference/agg/waiting_requests": float(waiting_requests),
        }


class TimestampSequence:
    def __init__(self, clock: Callable[[], float] = time.time):
        self.clock = clock
        self.previous = 0.0

    def next(self) -> float:
        self.previous = max(self.clock(), self.previous + 1e-6)
        return self.previous


class WorkerDaemon:
    def __init__(
        self,
        config: WorkerConfig,
        registration: WorkerRegistration,
        client: CoordinatorClient,
        spool: WorkerSpool,
        executor: AssignmentExecutor,
        *,
        timestamp_sequence: TimestampSequence | None = None,
        request_id_factory: Callable[[], str] = lambda: f"request-{uuid.uuid4().hex}",
        loaded_policy_ids: Callable[[], tuple[str, ...]] = lambda: (),
        policy_runtime: PolicyRuntime | None = None,
    ):
        self.config = config
        self.registration = registration
        self.client = client
        self.spool = spool
        self.executor = executor
        self.timestamps = timestamp_sequence or TimestampSequence()
        self.request_id_factory = request_id_factory
        self.policy_runtime = policy_runtime
        self.loaded_policy_ids = policy_runtime.loaded_policy_ids if policy_runtime is not None else loaded_policy_ids
        self.stop_event = asyncio.Event()
        self.lease_lock = asyncio.Lock()
        self.active: dict[str, ActiveAssignment] = {}
        self._entry_events: dict[str, asyncio.Event] = {}
        self._upload_claims: set[str] = set()
        self._upload_claim_lock = asyncio.Lock()
        self._server_time_offset = 0.0
        self.inference_metrics = InferenceMetrics()

    async def run(self) -> None:
        try:
            if self.policy_runtime is not None:
                await self.policy_runtime.start(self.stop_event)
            await self._run_control_plane()
        finally:
            if self.policy_runtime is not None:
                await self.policy_runtime.stop()

    async def _run_control_plane(self) -> None:
        await self._register()
        heartbeat = asyncio.create_task(self._heartbeat_loop(), name="worker-heartbeat")
        watchdog = asyncio.create_task(self._lease_watchdog_loop(), name="worker-lease-watchdog")
        uploaders = [
            asyncio.create_task(self._upload_loop(), name=f"worker-uploader-{index}")
            for index in range(self.config.result_upload_concurrency)
        ]
        policy_monitor = (
            asyncio.create_task(self.policy_runtime.monitor(), name="worker-policy-runtime")
            if self.policy_runtime is not None
            else None
        )
        slots = [
            asyncio.create_task(self._slot_loop(), name=f"worker-slot-{index}")
            for index in range(self.config.execution_slots)
        ]
        stop_waiter = asyncio.create_task(self.stop_event.wait(), name="worker-stop")
        tasks = [heartbeat, watchdog, *uploaders, *slots]
        if policy_monitor is not None:
            tasks.append(policy_monitor)
        try:
            done, _ = await asyncio.wait([stop_waiter, *tasks], return_when=asyncio.FIRST_COMPLETED)
            failed = next((task for task in done if task is not stop_waiter), None)
            if failed is not None:
                await failed
            for active in self.active.values():
                active.cancel_event.set()
            if slots:
                done_slots, _ = await asyncio.wait(slots, timeout=self.config.shutdown_grace_seconds)
                for task in done_slots:
                    await task
        finally:
            self.stop_event.set()
            stop_waiter.cancel()
            for active in self.active.values():
                active.cancel_event.set()
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(stop_waiter, *tasks, return_exceptions=True)

    def stop(self) -> None:
        self.stop_event.set()

    async def _register(self) -> None:
        delay = self.config.retry_min_seconds
        while not self.stop_event.is_set():
            try:
                request_started_at = self.timestamps.clock()
                response = await self.client.register(self.registration)
                if (
                    response.worker_id != self.registration.worker_id
                    or response.worker_session_id != self.registration.worker_session_id
                ):
                    raise CoordinatorProtocolError("registration response identity does not match")
                self._update_server_time(response.server_time, request_started_at)
                return
            except (httpx.TransportError, httpx.TimeoutException, CoordinatorProtocolError):
                await self._sleep(delay)
                delay = min(delay * 2, self.config.retry_max_seconds)
            except CoordinatorAPIError as error:
                if not error.retryable:
                    raise
                await self._sleep(error.retry_after if error.retry_after is not None else delay)
                delay = min(delay * 2, self.config.retry_max_seconds)
        raise asyncio.CancelledError

    async def _slot_loop(self) -> None:
        while not self.stop_event.is_set():
            if len(self.spool.entries()) + self.config.execution_slots > self.config.spool_max_entries:
                await self._sleep(self.config.retry_min_seconds)
                continue
            leases = await self._acquire_leases()
            if not leases:
                await self._sleep(self.config.retry_min_seconds)
                continue
            actives = []
            for lease in leases:
                self._validate_lease(lease)
                if lease.expires_at <= self._server_now():
                    continue
                active = ActiveAssignment(lease, asyncio.Event(), lease.expires_at)
                if self.stop_event.is_set():
                    active.cancel_event.set()
                if lease.lease_id in self.active:
                    raise RuntimeError("coordinator returned an already active lease")
                self.active[lease.lease_id] = active
                actives.append(active)
            if not actives:
                continue
            try:
                try:
                    execution_started_at = time.monotonic()
                    if self.policy_runtime is None:
                        envelopes = await self._execute_group(actives)
                    else:
                        async with self.policy_runtime.acquire(actives[0].lease.assignment.policy):
                            envelopes = await self._execute_group(actives)
                    execution_seconds = time.monotonic() - execution_started_at
                except asyncio.CancelledError:
                    if all(active.cancel_event.is_set() for active in actives) and not self.stop_event.is_set():
                        continue
                    raise
                except Exception as error:
                    if getattr(error, "worker_fatal", False):
                        raise
                    envelopes = [self._failure_envelope(active, error) for active in actives]
                    execution_seconds = 0.0
                self.inference_metrics.record_batch(len(actives))
                events = []
                for active, envelope in zip(actives, envelopes, strict=True):
                    if envelope is None:
                        continue
                    self.inference_metrics.record(active.lease, envelope, execution_seconds)
                    self._validate_terminal_envelope(active.lease, envelope)
                    entry = self.spool.publish(envelope)
                    events.append(self._entry_events.setdefault(entry.digest, asyncio.Event()))
                while events and not all(event.is_set() for event in events):
                    await asyncio.gather(
                        *(self._wait_for_event(event, self.config.heartbeat_interval_seconds) for event in events)
                    )
            finally:
                for active in actives:
                    self.active.pop(active.lease.lease_id, None)

    async def _acquire_lease(self) -> AssignmentLease | None:
        leases = await self._acquire_leases(max_slots=1)
        if len(leases) > 1:
            raise RuntimeError("single-lease acquisition received a lease group")
        return leases[0] if leases else None

    async def _acquire_leases(self, *, max_slots: int | None = None) -> tuple[AssignmentLease, ...]:
        async with self.lease_lock:
            free_slots = max(0, self.config.execution_slots - len(self.active))
            if max_slots is not None:
                free_slots = min(free_slots, max_slots)
            if free_slots < 1:
                return ()
            request = LeaseRequest(
                request_id=self.request_id_factory(),
                worker_id=self.registration.worker_id,
                worker_session_id=self.registration.worker_session_id,
                sent_at=self.timestamps.next(),
                loaded_policy_ids=tuple(sorted(self.loaded_policy_ids())),
                environments=self.registration.capabilities.environments,
                available_slots=free_slots,
                wait_seconds=self.config.lease_wait_seconds,
            )
            delay = self.config.retry_min_seconds
            while not self.stop_event.is_set():
                try:
                    if free_slots > 1 and hasattr(self.client, "lease_group"):
                        leases = await self.client.lease_group(request)
                        if leases:
                            return leases
                    lease = await self.client.lease(request)
                    return () if lease is None else (lease,)
                except (httpx.TransportError, httpx.TimeoutException, CoordinatorProtocolError):
                    await self._sleep(delay)
                    delay = min(delay * 2, self.config.retry_max_seconds)
                except CoordinatorAPIError as error:
                    legacy_capacity_conflict = (
                        error.status_code == 409
                        and error.code == "conflict"
                        and error.message
                        in {
                            "requested slots exceed worker session capacity",
                            "worker session has no free assignment capacity",
                        }
                    )
                    if not error.retryable and not legacy_capacity_conflict:
                        raise
                    await self._sleep(error.retry_after if error.retry_after is not None else delay)
                    delay = min(delay * 2, self.config.retry_max_seconds)
        return ()

    async def _execute(self, active: ActiveAssignment) -> TerminalEnvelope:
        try:
            return await self.executor.execute(active.lease, active.cancel_event)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return self._failure_envelope(active, error)

    async def _execute_group(self, actives: Sequence[ActiveAssignment]) -> list[TerminalEnvelope | None]:
        if len(actives) == 1:
            return [await self._execute(actives[0])]
        execute_group = getattr(self.executor, "execute_group", None)
        if execute_group is not None and self._can_execute_as_group(actives):
            results = await execute_group([(active.lease, active.cancel_event) for active in actives])
        else:
            results = await asyncio.gather(*(self._execute(active) for active in actives), return_exceptions=True)
        envelopes: list[TerminalEnvelope | None] = []
        for active, result in zip(actives, results, strict=True):
            if isinstance(result, asyncio.CancelledError):
                if len(actives) == 1 and active.cancel_event.is_set() and not self.stop_event.is_set():
                    raise result
                if active.cancel_event.is_set():
                    envelopes.append(None)
                    continue
                raise result
            if isinstance(result, BaseException):
                if getattr(result, "worker_fatal", False):
                    raise result
                envelopes.append(self._failure_envelope(active, result))
            else:
                envelopes.append(result)
        return envelopes

    @staticmethod
    def _can_execute_as_group(actives: Sequence[ActiveAssignment]) -> bool:
        if len(actives) < 2:
            return False
        first = actives[0].lease.assignment
        common = (first.group_id, first.environment, first.task_data, first.sampling, first.policy)
        return all(
            (active.lease.assignment.group_id, active.lease.assignment.environment, active.lease.assignment.task_data,
             active.lease.assignment.sampling, active.lease.assignment.policy)
            == common
            for active in actives
        )

    def _failure_envelope(self, active: ActiveAssignment, error: Exception) -> FailureEnvelope:
        message = str(error) or type(error).__name__
        return FailureEnvelope(
            assignment_id=active.lease.assignment.assignment_id,
            attempt=active.lease.attempt,
            lease_id=active.lease.lease_id,
            worker_id=self.registration.worker_id,
            worker_session_id=self.registration.worker_session_id,
            failed_at=self.timestamps.next(),
            code=getattr(error, "code", "execution_failed"),
            message=message[:8192],
            retryable=getattr(error, "retryable", True),
        )

    def _validate_terminal_envelope(self, lease: AssignmentLease, envelope: TerminalEnvelope) -> None:
        expected = (
            lease.assignment.assignment_id,
            lease.attempt,
            lease.lease_id,
            self.registration.worker_id,
            self.registration.worker_session_id,
        )
        actual = (
            envelope.assignment_id,
            envelope.attempt,
            envelope.lease_id,
            envelope.worker_id,
            envelope.worker_session_id,
        )
        if actual != expected:
            raise ValueError("executor returned an envelope for a different lease identity")
        if hasattr(envelope, "requested_policy_id"):
            digest = policy_manifest_digest(lease.assignment.policy)
            if envelope.requested_policy_id != lease.assignment.policy.policy_id:
                raise ValueError("executor result policy ID does not match its assignment")
            if envelope.requested_policy_digest != digest:
                raise ValueError("executor result policy digest does not match its assignment")

    def _validate_lease(self, lease: AssignmentLease) -> None:
        if (
            lease.worker_id != self.registration.worker_id
            or lease.worker_session_id != self.registration.worker_session_id
        ):
            raise CoordinatorProtocolError("lease belongs to a different worker session")
        if lease.assignment.environment not in self.registration.capabilities.environments:
            raise CoordinatorProtocolError("lease requires an unsupported environment")
        if lease.assignment.policy.base_model != self.registration.capabilities.base_model:
            raise CoordinatorProtocolError("lease requires a different base model")

    async def _heartbeat_loop(self) -> None:
        while not self.stop_event.is_set():
            heartbeat = WorkerHeartbeat(
                worker_id=self.registration.worker_id,
                worker_session_id=self.registration.worker_session_id,
                sent_at=self.timestamps.next(),
                active_lease_ids=tuple(sorted(self.active)),
                loaded_policy_ids=tuple(sorted(self.loaded_policy_ids())),
            )
            try:
                request_started_at = self.timestamps.clock()
                response = await self.client.heartbeat(heartbeat)
            except (httpx.TransportError, httpx.TimeoutException, CoordinatorProtocolError):
                await self._sleep(self.config.retry_min_seconds)
                continue
            except CoordinatorAPIError as error:
                if not error.retryable:
                    raise
                await self._sleep(error.retry_after or self.config.retry_min_seconds)
                continue
            self._update_server_time(response.server_time, request_started_at)
            for renewal in response.renewals:
                active = self.active.get(renewal.lease_id)
                if active is not None:
                    if renewal.assignment_id != active.lease.assignment.assignment_id:
                        raise CoordinatorProtocolError("lease renewal assignment does not match its lease")
                    if renewal.expires_at < active.expires_at:
                        raise CoordinatorProtocolError("lease renewal shortened its expiry")
                    deadline = active.lease.assignment.deadline_at
                    if deadline is not None and renewal.expires_at > deadline:
                        raise CoordinatorProtocolError("lease renewal exceeds the assignment deadline")
                    active.expires_at = renewal.expires_at
            for lease_id in response.stop_lease_ids:
                active = self.active.get(lease_id)
                if active is not None:
                    active.cancel_event.set()
            await self._sleep(self.config.heartbeat_interval_seconds)

    async def _lease_watchdog_loop(self) -> None:
        interval = min(self.config.heartbeat_interval_seconds, 1.0)
        while not self.stop_event.is_set():
            now = self._server_now()
            for active in self.active.values():
                if active.expires_at <= now:
                    active.cancel_event.set()
            await self._sleep(interval)

    async def _upload_loop(self) -> None:
        delay = self.config.retry_min_seconds
        while not self.stop_event.is_set() or self.spool.entries():
            entry = await self._claim_upload_entry()
            if entry is None:
                delay = self.config.retry_min_seconds
                await self._sleep(delay)
                continue
            try:
                response = await self.client.submit(entry)
                self.spool.acknowledge(entry, response)
            except (httpx.TransportError, httpx.TimeoutException, CoordinatorProtocolError):
                try:
                    await asyncio.sleep(delay)
                finally:
                    await self._release_upload_entry(entry)
                delay = min(delay * 2, self.config.retry_max_seconds)
                continue
            except CoordinatorAPIError as error:
                if error.retryable:
                    try:
                        await asyncio.sleep(error.retry_after if error.retry_after is not None else delay)
                    finally:
                        await self._release_upload_entry(entry)
                    delay = min(delay * 2, self.config.retry_max_seconds)
                    continue
                if error.status_code in {409, 413, 415, 422}:
                    self.spool.reject(entry)
                else:
                    await self._release_upload_entry(entry)
                    raise
            event = self._entry_events.get(entry.digest)
            if event is not None:
                event.set()
            await self._release_upload_entry(entry)
            delay = self.config.retry_min_seconds

    async def _claim_upload_entry(self):
        async with self._upload_claim_lock:
            for entry in self.spool.entries():
                if entry.digest not in self._upload_claims:
                    self._upload_claims.add(entry.digest)
                    return entry
        return None

    async def _release_upload_entry(self, entry) -> None:
        async with self._upload_claim_lock:
            self._upload_claims.discard(entry.digest)

    def _update_server_time(self, server_time: float, request_started_at: float) -> None:
        # The response timestamp was generated after the request started. Using
        # the start time gives an upper bound on coordinator time at receipt,
        # so network delay can only cancel work early, never extend it stale.
        self._server_time_offset = server_time - request_started_at

    def _server_now(self) -> float:
        return self.timestamps.clock() + self._server_time_offset

    async def _sleep(self, seconds: float) -> None:
        if seconds <= 0:
            return
        try:
            await asyncio.wait_for(self.stop_event.wait(), timeout=seconds)
        except TimeoutError:
            pass

    async def _wait_for_event(self, event: asyncio.Event, timeout: float) -> None:
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except TimeoutError:
            pass


def _episode_output_tokens(envelope: TerminalEnvelope) -> int:
    episode = getattr(envelope, "episode", None)
    traces = getattr(episode, "traces", ()) if episode is not None else ()
    return sum(int(getattr(trace, "num_output_tokens", 0) or 0) for trace in traces)


def _episode_generation_seconds(envelope: TerminalEnvelope) -> float:
    episode = getattr(envelope, "episode", None)
    traces = getattr(episode, "traces", ()) if episode is not None else ()
    total = 0.0
    for trace in traces:
        timing = getattr(trace, "timing", None)
        generation = getattr(timing, "generation", None)
        total += float(getattr(generation, "duration", 0.0) or 0.0)
    return total


def build_registration(
    config: WorkerConfig,
    worker_id: str,
    worker_session_id: str,
    *,
    clock: Callable[[], float] = time.time,
    gpu_count: int | None = None,
) -> WorkerRegistration:
    discovered_gpu_count = torch.cuda.device_count() if gpu_count is None else gpu_count
    if discovered_gpu_count < 1:
        raise RuntimeError("worker requires at least one visible GPU")
    base_model = BaseModelIdentity.model_validate(config.base_model.model_dump(mode="python"))
    for environment in config.environments:
        installed_revision = importlib.metadata.version(environment.package)
        if installed_revision != environment.revision:
            raise RuntimeError(
                f"environment package {environment.package!r} is {installed_revision}, expected {environment.revision}"
            )
    runtime = RuntimeIdentity(
        aether_rl_version=importlib.metadata.version("aether-rl"),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=importlib.metadata.version("transformers"),
        vllm_version=importlib.metadata.version("vllm"),
        cuda_version=torch.version.cuda,
    )
    capabilities = WorkerCapabilities(
        base_model=base_model,
        runtime=runtime,
        environments=tuple(
            EnvironmentIdentity(id=environment.id, revision=environment.revision) for environment in config.environments
        ),
        max_concurrent_assignments=config.execution_slots,
        gpu_count=discovered_gpu_count,
        tensor_parallel_size=config.tensor_parallel_size,
        labels=config.labels,
    )
    return WorkerRegistration(
        worker_id=worker_id,
        worker_session_id=worker_session_id,
        registered_at=clock(),
        capabilities=capabilities,
    )


async def run_worker(config: WorkerConfig, executor: AssignmentExecutor | None = None) -> None:
    if executor is None:
        from .executor import VerifiersAssignmentExecutor

        executor = VerifiersAssignmentExecutor(config)
    token = os.environ.get("AETHER_COORDINATOR_TOKEN")
    if not token:
        raise RuntimeError("AETHER_COORDINATOR_TOKEN is required")
    state = WorkerState(config.state_dir)
    try:
        from .identity import discover_base_model_identity
        from .policy_runtime import WorkerPolicyRuntime

        await asyncio.to_thread(discover_base_model_identity, config)
        worker_id = state.load_or_create_worker_id()
        session_id = f"session-{uuid.uuid4().hex}"
        registration = build_registration(config, worker_id, session_id)
        spool = WorkerSpool(state)
        client = CoordinatorClient(
            str(config.coordinator_url),
            token,
            timeout_seconds=config.request_timeout_seconds,
            result_compression=config.result_compression,
        )
        policy_runtime = WorkerPolicyRuntime(config, state, client)
        daemon = WorkerDaemon(config, registration, client, spool, executor, policy_runtime=policy_runtime)
        loop = asyncio.get_running_loop()
        for signal_name in ("SIGINT", "SIGTERM"):
            import signal

            loop.add_signal_handler(getattr(signal, signal_name), daemon.stop)
        try:
            await daemon.run()
        finally:
            await client.close()
    finally:
        state.close()
