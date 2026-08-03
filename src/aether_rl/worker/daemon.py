from __future__ import annotations

import asyncio
import importlib.metadata
import json
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
    BaseModelIdentity,
    InferenceExchangeRequest,
    InferenceLease,
    InferenceReply,
    LeaseRequest,
    RuntimeIdentity,
    WorkerCapabilities,
    WorkerHeartbeat,
    WorkerRegistration,
)

from .client import CoordinatorAPIError, CoordinatorClient, CoordinatorProtocolError
from .state import WorkerState


class PolicyRuntime(Protocol):
    async def start(self, stop_event: asyncio.Event) -> None: ...
    async def stop(self) -> None: ...
    async def monitor(self) -> None: ...
    def loaded_policy_ids(self) -> tuple[str, ...]: ...
    def acquire(self, manifest) -> AbstractAsyncContextManager[str]: ...


@dataclass
class ActiveInference:
    lease: InferenceLease
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
        *,
        timestamp_sequence: TimestampSequence | None = None,
        request_id_factory: Callable[[], str] = lambda: f"request-{uuid.uuid4().hex}",
        loaded_policy_ids: Callable[[], tuple[str, ...]] = lambda: (),
        policy_runtime: PolicyRuntime | None = None,
        inference_client: httpx.AsyncClient | None = None,
    ):
        self.config = config
        self.registration = registration
        self.client = client
        self.timestamps = timestamp_sequence or TimestampSequence()
        self.request_id_factory = request_id_factory
        self.policy_runtime = policy_runtime
        self.loaded_policy_ids = policy_runtime.loaded_policy_ids if policy_runtime is not None else loaded_policy_ids
        self.stop_event = asyncio.Event()
        self.lease_lock = asyncio.Lock()
        self.active: dict[str, ActiveInference] = {}
        self._server_time_offset = 0.0
        self._owns_inference_client = inference_client is None
        self.inference_client = inference_client or httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{config.inference_port}",
            timeout=None,
        )

    async def run(self) -> None:
        try:
            if self.policy_runtime is not None:
                await self.policy_runtime.start(self.stop_event)
            await self._register()
            tasks = [
                asyncio.create_task(self._heartbeat_loop(), name="worker-heartbeat"),
                asyncio.create_task(self._lease_watchdog_loop(), name="worker-lease-watchdog"),
                *(
                    asyncio.create_task(self._slot_loop(), name=f"worker-inference-slot-{index}")
                    for index in range(self.config.inference_slots)
                ),
            ]
            if self.policy_runtime is not None:
                tasks.append(asyncio.create_task(self.policy_runtime.monitor(), name="worker-policy-runtime"))
            stop_waiter = asyncio.create_task(self.stop_event.wait(), name="worker-stop")
            try:
                done, _ = await asyncio.wait([stop_waiter, *tasks], return_when=asyncio.FIRST_COMPLETED)
                failed = next((task for task in done if task is not stop_waiter), None)
                if failed is not None:
                    await failed
            finally:
                self.stop_event.set()
                stop_waiter.cancel()
                for active in self.active.values():
                    active.cancel_event.set()
                for task in tasks:
                    task.cancel()
                await asyncio.gather(stop_waiter, *tasks, return_exceptions=True)
        finally:
            if self.policy_runtime is not None:
                await self.policy_runtime.stop()
            if self._owns_inference_client:
                await self.inference_client.aclose()

    def stop(self) -> None:
        self.stop_event.set()

    async def _register(self) -> None:
        delay = self.config.retry_min_seconds
        while not self.stop_event.is_set():
            try:
                started = self.timestamps.clock()
                response = await self.client.register(self.registration)
                if (response.worker_id, response.worker_session_id) != (
                    self.registration.worker_id,
                    self.registration.worker_session_id,
                ):
                    raise CoordinatorProtocolError("registration response identity does not match")
                self._update_server_time(response.server_time, started)
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
            lease = await self._acquire_lease()
            if lease is None:
                await self._sleep(self.config.retry_min_seconds)
                continue
            self._validate_lease(lease)
            if lease.expires_at <= self._server_now():
                continue
            active = ActiveInference(lease, asyncio.Event(), lease.expires_at)
            self.active[lease.lease_id] = active
            try:
                if self.policy_runtime is None:
                    await self._serve_lease(active)
                else:
                    async with self.policy_runtime.acquire(lease.policy):
                        await self._serve_lease(active)
            finally:
                self.active.pop(lease.lease_id, None)

    async def _acquire_lease(self) -> InferenceLease | None:
        async with self.lease_lock:
            free_slots = self.config.inference_slots - len(self.active)
            if free_slots < 1:
                return None
            request = LeaseRequest(
                request_id=self.request_id_factory(),
                worker_id=self.registration.worker_id,
                worker_session_id=self.registration.worker_session_id,
                sent_at=self.timestamps.next(),
                loaded_policy_ids=tuple(sorted(self.loaded_policy_ids())),
                available_slots=free_slots,
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
                    if not error.retryable:
                        raise
                    await self._sleep(error.retry_after if error.retry_after is not None else delay)
                    delay = min(delay * 2, self.config.retry_max_seconds)
        return None

    async def _serve_lease(self, active: ActiveInference) -> None:
        reply = None
        while not self.stop_event.is_set() and not active.cancel_event.is_set():
            exchange = InferenceExchangeRequest(
                worker_id=self.registration.worker_id,
                worker_session_id=self.registration.worker_session_id,
                lease_id=active.lease.lease_id,
                reply=reply,
                wait_seconds=self.config.lease_wait_seconds,
            )
            response = await self._retry_exchange(exchange)
            reply = None
            if response.action == "stop":
                return
            if response.action == "wait":
                continue
            request = response.request
            if request is None:
                raise CoordinatorProtocolError("request exchange action has no request")
            try:
                local_request = asyncio.create_task(
                    self.inference_client.request(
                        request.method,
                        request.path,
                        headers=request.headers,
                        content=request.body,
                    )
                )
                cancelled = asyncio.create_task(active.cancel_event.wait())
                try:
                    done, _ = await asyncio.wait({local_request, cancelled}, return_when=asyncio.FIRST_COMPLETED)
                    if cancelled in done:
                        local_request.cancel()
                        await asyncio.gather(local_request, return_exceptions=True)
                        return
                    local = await local_request
                finally:
                    cancelled.cancel()
                    await asyncio.gather(cancelled, return_exceptions=True)
                body = local.content
                if len(body) > self.config.inference_body_limit_bytes:
                    raise ValueError("local inference reply is too large")
                headers = {
                    key: value
                    for key, value in local.headers.items()
                    if key.lower() in {"content-type", "retry-after", "x-request-id"}
                }
                reply = InferenceReply(
                    request_id=request.request_id,
                    status_code=local.status_code,
                    headers=headers,
                    body=body,
                )
            except (httpx.HTTPError, ValueError) as error:
                reply = InferenceReply(
                    request_id=request.request_id,
                    status_code=502,
                    headers={"content-type": "application/json"},
                    body=json.dumps({"error": {"message": str(error), "type": "worker_inference_error"}}).encode(),
                )

    async def _retry_exchange(self, request: InferenceExchangeRequest):
        delay = self.config.retry_min_seconds
        while not self.stop_event.is_set():
            try:
                return await self.client.inference_exchange(request)
            except (httpx.TransportError, httpx.TimeoutException, CoordinatorProtocolError):
                await self._sleep(delay)
                delay = min(delay * 2, self.config.retry_max_seconds)
            except CoordinatorAPIError as error:
                if not error.retryable:
                    raise
                await self._sleep(error.retry_after if error.retry_after is not None else delay)
                delay = min(delay * 2, self.config.retry_max_seconds)
        raise asyncio.CancelledError

    def _validate_lease(self, lease: InferenceLease) -> None:
        if (lease.worker_id, lease.worker_session_id) != (
            self.registration.worker_id,
            self.registration.worker_session_id,
        ):
            raise CoordinatorProtocolError("lease belongs to a different worker session")
        if lease.policy.base_model != self.registration.capabilities.base_model:
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
                started = self.timestamps.clock()
                response = await self.client.heartbeat(heartbeat)
            except (httpx.TransportError, httpx.TimeoutException, CoordinatorProtocolError):
                await self._sleep(self.config.retry_min_seconds)
                continue
            except CoordinatorAPIError as error:
                if not error.retryable:
                    raise
                await self._sleep(error.retry_after or self.config.retry_min_seconds)
                continue
            self._update_server_time(response.server_time, started)
            for renewal in response.renewals:
                active = self.active.get(renewal.lease_id)
                if active is not None:
                    if renewal.assignment_id != active.lease.assignment_id:
                        raise CoordinatorProtocolError("lease renewal assignment does not match its lease")
                    active.expires_at = renewal.expires_at
            for lease_id in response.stop_lease_ids:
                active = self.active.get(lease_id)
                if active is not None:
                    active.cancel_event.set()
            await self._sleep(self.config.heartbeat_interval_seconds)

    async def _lease_watchdog_loop(self) -> None:
        while not self.stop_event.is_set():
            now = self._server_now()
            for active in self.active.values():
                if active.expires_at <= now:
                    active.cancel_event.set()
            await self._sleep(min(self.config.heartbeat_interval_seconds, 1.0))

    def _update_server_time(self, server_time: float, request_started_at: float) -> None:
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
    runtime = RuntimeIdentity(
        aether_rl_version=importlib.metadata.version("aether-rl"),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        transformers_version=importlib.metadata.version("transformers"),
        vllm_version=importlib.metadata.version("vllm"),
        cuda_version=torch.version.cuda,
    )
    capabilities = WorkerCapabilities(
        base_model=BaseModelIdentity.model_validate(config.base_model.model_dump(mode="python")),
        runtime=runtime,
        inference_slots=config.inference_slots,
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


async def run_worker(config: WorkerConfig) -> None:
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
        client = CoordinatorClient(
            str(config.coordinator_url),
            token,
            timeout_seconds=config.request_timeout_seconds,
        )
        policy_runtime = WorkerPolicyRuntime(config, state, client)
        daemon = WorkerDaemon(config, registration, client, policy_runtime=policy_runtime)
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
