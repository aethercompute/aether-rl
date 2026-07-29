import asyncio
import importlib.metadata
from pathlib import Path

import httpx
import pytest
from verifiers.v1.types import SamplingConfig

from aether_rl.configs.worker import WorkerConfig, WorkerEnvironmentConfig
from aether_rl.protocol import (
    AssignmentLease,
    SubmissionResponse,
    WorkerHeartbeatResponse,
    WorkerRegistrationResponse,
    sha256_digest,
)
from aether_rl.worker.client import CoordinatorAPIError
from aether_rl.worker.daemon import ActiveAssignment, TimestampSequence, WorkerDaemon, build_registration
from aether_rl.worker.spool import SpoolCorruptionError, WorkerSpool, WorkerState, WorkerStateError
from tests.unit.coordinator.test_database import (
    assignments,
    base_model,
    base_policy,
    failure_envelope,
    registration,
    result_envelope,
)


def worker_config(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "coordinator_url": "https://coordinator.example.com",
            "state_dir": tmp_path / "worker",
            "base_model": base_model().model_dump(mode="python"),
            "environments": [
                {"id": "env", "package": "verifiers", "revision": "1", "config": {"taskset": {"id": "env"}}}
            ],
            "execution_slots": 1,
            "tensor_parallel_size": 1,
            "heartbeat_interval_seconds": 0.01,
            "lease_wait_seconds": 0,
            "request_timeout_seconds": 1,
            "retry_min_seconds": 0.001,
            "retry_max_seconds": 0.01,
            "shutdown_grace_seconds": 1,
        }
    )


def assignment_lease() -> AssignmentLease:
    assignment = assignments(base_policy())[0].model_copy(
        update={"sampling": SamplingConfig(temperature=1, max_tokens=8)}
    )
    return AssignmentLease(
        lease_id="lease-1",
        attempt=1,
        worker_id="worker-1",
        worker_session_id="session-1",
        issued_at=3,
        expires_at=100,
        assignment=assignment,
    )


def test_worker_identity_lock_spool_recovery_acknowledgement_and_quarantine(tmp_path: Path):
    state_path = tmp_path / "missing-parent" / "state"
    state = WorkerState(state_path)
    worker_id = state.load_or_create_worker_id()
    assert state.load_or_create_worker_id() == worker_id
    with pytest.raises(WorkerStateError, match="already owned"):
        WorkerState(state_path)

    spool = WorkerSpool(state)
    entry = spool.publish(failure_envelope(assignment_lease()))
    assert spool.publish(entry.envelope) == entry
    assert spool.entries() == (entry,)
    spool.acknowledge(
        entry,
        SubmissionResponse(
            assignment_id=entry.envelope.assignment_id,
            envelope_digest=entry.digest,
            duplicate=False,
            terminal=False,
        ),
    )
    assert spool.entries() == ()
    state.close()

    with WorkerState(state_path) as reopened:
        assert reopened.load_or_create_worker_id() == worker_id
        spool = WorkerSpool(reopened)
        corrupt = spool.pending / f"{'0' * 64}.failure.json"
        corrupt.write_bytes(b"not-json")
        with pytest.raises(SpoolCorruptionError, match="invalid pending"):
            spool.entries()
        assert not corrupt.exists()
        assert (spool.rejected / corrupt.name).read_bytes() == b"not-json"
        pending = spool.publish(result_envelope(assignment_lease()))

    with WorkerState(state_path) as restarted:
        recovered = WorkerSpool(restarted).entries()
        assert len(recovered) == 1
        assert recovered[0].body == pending.body


def test_capability_discovery_verifies_environment_revision(tmp_path: Path):
    config = worker_config(tmp_path).model_copy(
        update={
            "environments": [
                WorkerEnvironmentConfig(
                    id="env",
                    package="verifiers",
                    revision=importlib.metadata.version("verifiers"),
                    config={"taskset": {"id": "env"}},
                )
            ]
        }
    )
    discovered = build_registration(config, "worker-1", "session-1", gpu_count=1)
    assert discovered.capabilities.environments[0].revision == importlib.metadata.version("verifiers")
    mismatched = config.model_copy(
        update={"environments": [config.environments[0].model_copy(update={"revision": "wrong"})]}
    )
    with pytest.raises(RuntimeError, match="expected wrong"):
        build_registration(mismatched, "worker-1", "session-1", gpu_count=1)


def test_failure_envelope_preserves_terminal_execution_errors(tmp_path: Path):
    class TerminalExecutionError(RuntimeError):
        code = "result_too_large"
        retryable = False

    config = worker_config(tmp_path)
    with WorkerState(config.state_dir) as state:
        daemon = WorkerDaemon(
            config, registration(), FakeCoordinatorClient(assignment_lease()), WorkerSpool(state), FakeExecutor()
        )
        envelope = daemon._failure_envelope(
            ActiveAssignment(assignment_lease(), asyncio.Event(), 100),
            TerminalExecutionError("too large"),
        )
    assert envelope.code == "result_too_large"
    assert envelope.retryable is False


class FakeExecutor:
    async def execute(self, lease, cancel_event):
        assert not cancel_event.is_set()
        return result_envelope(lease)


class FakeCoordinatorClient:
    def __init__(self, lease: AssignmentLease):
        self.offered = lease
        self.daemon: WorkerDaemon | None = None
        self.lease_requests = []
        self.submissions = []
        self.lease_calls = 0
        self.submit_calls = 0

    async def register(self, request):
        return WorkerRegistrationResponse(
            worker_id=request.worker_id,
            worker_session_id=request.worker_session_id,
            created=True,
            server_time=1,
        )

    async def heartbeat(self, request):
        return WorkerHeartbeatResponse(server_time=1)

    async def lease(self, request):
        self.lease_requests.append(request)
        self.lease_calls += 1
        if self.lease_calls == 1:
            raise httpx.ReadTimeout("lost lease response")
        if self.lease_calls == 2:
            return self.offered
        return None

    async def submit(self, entry):
        self.submissions.append(entry.body)
        self.submit_calls += 1
        if self.submit_calls == 1:
            raise httpx.ReadTimeout("lost result response")
        assert self.daemon is not None
        self.daemon.stop()
        return SubmissionResponse(
            assignment_id=entry.envelope.assignment_id,
            envelope_digest=sha256_digest(entry.body),
            duplicate=True,
            terminal=True,
        )


@pytest.mark.asyncio
async def test_daemon_retries_exact_lease_and_result_bytes_then_stops_cleanly(tmp_path: Path):
    config = worker_config(tmp_path)
    worker_registration = registration()
    lease = assignment_lease()
    client = FakeCoordinatorClient(lease)
    with WorkerState(config.state_dir) as state:
        spool = WorkerSpool(state)
        daemon = WorkerDaemon(
            config,
            worker_registration,
            client,  # type: ignore[arg-type]
            spool,
            FakeExecutor(),
            timestamp_sequence=TimestampSequence(lambda: 10),
            request_id_factory=lambda: "request-1",
        )
        client.daemon = daemon
        await daemon.run()
        assert len(client.lease_requests) >= 2
        assert client.lease_requests[0] == client.lease_requests[1]
        assert client.lease_requests[0].sent_at >= 10
        assert client.submissions[0] == client.submissions[1]
        assert spool.entries() == ()
        assert daemon.active == {}


class CancellationExecutor:
    def __init__(self):
        self.cancelled = False

    async def execute(self, lease, cancel_event):
        await cancel_event.wait()
        self.cancelled = True
        raise RuntimeError("coordinator cancelled assignment")


class CancelledErrorExecutor:
    def __init__(self):
        self.cancelled = False

    async def execute(self, lease, cancel_event):
        await cancel_event.wait()
        self.cancelled = True
        raise asyncio.CancelledError


class CancellationClient(FakeCoordinatorClient):
    async def lease(self, request):
        self.lease_requests.append(request)
        if len(self.lease_requests) == 1:
            return self.offered
        return None

    async def heartbeat(self, request):
        return WorkerHeartbeatResponse(server_time=1, stop_lease_ids=request.active_lease_ids)

    async def submit(self, entry):
        assert self.daemon is not None
        self.daemon.stop()
        return SubmissionResponse(
            assignment_id=entry.envelope.assignment_id,
            envelope_digest=entry.digest,
            duplicate=False,
            terminal=True,
        )


@pytest.mark.asyncio
async def test_heartbeat_cancellation_stops_executor_and_spools_failure(tmp_path: Path):
    config = worker_config(tmp_path)
    client = CancellationClient(assignment_lease())
    executor = CancellationExecutor()
    with WorkerState(config.state_dir) as state:
        spool = WorkerSpool(state)
        daemon = WorkerDaemon(config, registration(), client, spool, executor)
        client.daemon = daemon
        await daemon.run()
        assert executor.cancelled
        assert spool.entries() == ()


@pytest.mark.asyncio
async def test_heartbeat_cancelled_error_does_not_stop_worker(tmp_path: Path):
    config = worker_config(tmp_path)
    client = CancellationClient(assignment_lease())
    executor = CancelledErrorExecutor()
    with WorkerState(config.state_dir) as state:
        spool = WorkerSpool(state)
        daemon = WorkerDaemon(config, registration(), client, spool, executor)
        client.daemon = daemon
        task = asyncio.create_task(daemon.run())
        while not executor.cancelled:
            await asyncio.sleep(0)
        daemon.stop()
        await task
        assert len(client.lease_requests) > 1
        assert spool.entries() == ()


@pytest.mark.asyncio
async def test_multi_slot_requests_reserve_one_slot_and_auth_failure_preserves_spool(tmp_path: Path):
    config = worker_config(tmp_path).model_copy(update={"execution_slots": 2, "spool_max_entries": 2})

    class Client(FakeCoordinatorClient):
        async def lease(self, request):
            self.lease_requests.append(request)
            return None

        async def submit(self, entry):
            raise CoordinatorAPIError(401, "unauthorized", "authentication required")

    client = Client(assignment_lease())
    with WorkerState(config.state_dir) as state:
        spool = WorkerSpool(state)
        entry = spool.publish(failure_envelope(assignment_lease()))
        request_ids = iter(("request-a", "request-b"))
        daemon = WorkerDaemon(
            config,
            registration(),
            client,  # type: ignore[arg-type]
            spool,
            FakeExecutor(),
            request_id_factory=lambda: next(request_ids),
        )
        assert await daemon._acquire_lease() is None
        assert await daemon._acquire_lease() is None
        assert [request.available_slots for request in client.lease_requests] == [1, 1]
        with pytest.raises(CoordinatorAPIError, match="unauthorized"):
            await daemon._upload_loop()
        assert spool.entries() == (entry,)
