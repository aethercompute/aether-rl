from __future__ import annotations

import asyncio
import importlib.metadata
import os
import platform
import time
import uuid
from collections.abc import Callable
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
        self._server_time_offset = 0.0

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
        uploader = asyncio.create_task(self._upload_loop(), name="worker-uploader")
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
        tasks = [heartbeat, watchdog, uploader, *slots]
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
            lease = await self._acquire_lease()
            if lease is None:
                await self._sleep(self.config.retry_min_seconds)
                continue
            self._validate_lease(lease)
            if lease.expires_at <= self._server_now():
                continue
            active = ActiveAssignment(lease, asyncio.Event(), lease.expires_at)
            if self.stop_event.is_set():
                active.cancel_event.set()
            if lease.lease_id in self.active:
                raise RuntimeError("coordinator returned an already active lease")
            self.active[lease.lease_id] = active
            try:
                try:
                    if self.policy_runtime is None:
                        envelope = await self._execute(active)
                    else:
                        async with self.policy_runtime.acquire(lease.assignment.policy):
                            envelope = await self._execute(active)
                except asyncio.CancelledError:
                    if active.cancel_event.is_set() and not self.stop_event.is_set():
                        continue
                    raise
                except Exception as error:
                    if getattr(error, "worker_fatal", False):
                        raise
                    envelope = self._failure_envelope(active, error)
                self._validate_terminal_envelope(lease, envelope)
                entry = self.spool.publish(envelope)
                event = self._entry_events.setdefault(entry.digest, asyncio.Event())
                while not event.is_set():
                    await self._wait_for_event(event, self.config.heartbeat_interval_seconds)
            finally:
                self.active.pop(lease.lease_id, None)

    async def _acquire_lease(self) -> AssignmentLease | None:
        async with self.lease_lock:
            request = LeaseRequest(
                request_id=self.request_id_factory(),
                worker_id=self.registration.worker_id,
                worker_session_id=self.registration.worker_session_id,
                sent_at=self.timestamps.next(),
                loaded_policy_ids=tuple(sorted(self.loaded_policy_ids())),
                environments=self.registration.capabilities.environments,
                available_slots=1,
                wait_seconds=self.config.lease_wait_seconds,
            )
            delay = self.config.retry_min_seconds
            while not self.stop_event.is_set():
                try:
                    return await self.client.lease(request)
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
        return None

    async def _execute(self, active: ActiveAssignment) -> TerminalEnvelope:
        try:
            return await self.executor.execute(active.lease, active.cancel_event)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return self._failure_envelope(active, error)

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
            entries = self.spool.entries()
            if not entries:
                delay = self.config.retry_min_seconds
                await self._sleep(delay)
                continue
            entry = entries[0]
            try:
                response = await self.client.submit(entry)
                self.spool.acknowledge(entry, response)
            except (httpx.TransportError, httpx.TimeoutException, CoordinatorProtocolError):
                await asyncio.sleep(delay)
                delay = min(delay * 2, self.config.retry_max_seconds)
                continue
            except CoordinatorAPIError as error:
                if error.retryable:
                    await asyncio.sleep(error.retry_after if error.retry_after is not None else delay)
                    delay = min(delay * 2, self.config.retry_max_seconds)
                    continue
                if error.status_code in {409, 413, 415, 422}:
                    self.spool.reject(entry)
                else:
                    raise
            event = self._entry_events.get(entry.digest)
            if event is not None:
                event.set()
            delay = self.config.retry_min_seconds

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
