import asyncio
from pathlib import Path

import httpx
import pytest
from verifiers.v1.types import SamplingConfig

from aether_rl.configs.algorithm import GRPOAlgoConfig
from aether_rl.coordinator import (
    RemoteResultProcessor,
    ResultProcessingSource,
    create_coordinator_app,
    decode_training_batch,
)
from aether_rl.coordinator.environments import EnvironmentSourceSpec
from aether_rl.orchestrator.algo import GRPOAlgorithm
from aether_rl.protocol import (
    EnvironmentIdentity,
    LeaseRequest,
    WorkerHeartbeat,
    canonical_json_bytes,
    result_envelope_bytes,
)
from aether_rl.worker.client import CoordinatorClient
from aether_rl.worker.daemon import TimestampSequence, WorkerDaemon
from aether_rl.worker.spool import WorkerSpool, WorkerState
from tests.unit.coordinator.test_database import (
    FakeClock,
    assignments,
    base_policy,
    capabilities,
    registration,
)
from tests.unit.coordinator.test_results import result_envelope
from tests.unit.coordinator.test_scheduler import make_repository, publish_policy
from tests.unit.worker.test_worker import worker_config

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}", "Aether-Protocol-Version": "1"}
JSON_HEADERS = {**AUTH, "Content-Type": "application/json"}
MSGPACK_HEADERS = {**AUTH, "Content-Type": "application/msgpack"}


class StopAfterSubmitClient(CoordinatorClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.daemon: WorkerDaemon | None = None

    async def submit(self, entry):
        response = await super().submit(entry)
        if self.daemon is not None:
            self.daemon.stop()
        return response

    async def close(self) -> None:
        await self.client.aclose()


class RewardExecutor:
    def __init__(self, reward: float):
        self.reward = reward

    async def execute(self, lease, cancel_event):
        assert not cancel_event.is_set()
        return result_envelope(lease, self.reward)


def train_source(group_size: int = 2) -> EnvironmentSourceSpec:
    return EnvironmentSourceSpec(
        source_id="train-source",
        kind="train",
        environment=EnvironmentIdentity(id="env", revision="1"),
        tasks=({"idx": 0, "prompt": "prompt"},),
        sampling=SamplingConfig(temperature=1, max_tokens=8),
        group_size=group_size,
        max_attempts=2,
        assignment_timeout_seconds=100,
    )


@pytest.mark.asyncio
async def test_two_workers_complete_one_coordinator_group_and_emit_training_batch(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_scheduler_source(train_source())
        app = create_coordinator_app(repository, token=TOKEN, trainer_ready=lambda: True, lease_duration_seconds=10)
        transport = httpx.ASGITransport(app=app)
        daemons: list[WorkerDaemon] = []
        clients: list[StopAfterSubmitClient] = []
        states: list[WorkerState] = []
        try:
            for index, reward in enumerate((1.0, 0.0), start=1):
                worker_root = tmp_path / f"worker-{index}"
                worker_root.mkdir()
                config = worker_config(worker_root)
                state = WorkerState(config.state_dir)
                states.append(state)
                client = StopAfterSubmitClient(
                    "http://test",
                    TOKEN,
                    timeout_seconds=1,
                    client=httpx.AsyncClient(transport=transport, base_url="http://test"),
                )
                clients.append(client)
                daemon = WorkerDaemon(
                    config,
                    registration(worker_id=f"worker-{index}", session_id=f"session-{index}"),
                    client,
                    WorkerSpool(state),
                    RewardExecutor(reward),
                    timestamp_sequence=TimestampSequence(lambda: clock.now),
                    request_id_factory=lambda index=index: f"request-worker-{index}",
                )
                client.daemon = daemon
                daemons.append(daemon)

            await asyncio.wait_for(asyncio.gather(*(daemon.run() for daemon in daemons)), timeout=5)
            assert repository.connection.execute("SELECT COUNT(*) FROM accepted_results").fetchone()[0] == 2
            assert all(WorkerSpool(state).entries() == () for state in states)

            processor = RemoteResultProcessor(
                repository,
                (
                    ResultProcessingSource(
                        source_id="train-source",
                        environment=EnvironmentIdentity(id="env", revision="1"),
                        processing_id="grpo-v1",
                        algorithm=GRPOAlgorithm(GRPOAlgoConfig(), None),  # type: ignore[arg-type]
                    ),
                ),
                batch_size=2,
            )
            assert await processor.process_available() == (1, 1)
            batch = decode_training_batch(repository.training_batches()[0])
            assert batch.step == 1
            assert len(batch.examples) == 2
            assert {sample.behavior_policy_id for sample in batch.examples} == {base_policy().policy_id}
        finally:
            for daemon in daemons:
                daemon.stop()
            await asyncio.gather(*(client.close() for client in clients), return_exceptions=True)
            for state in states:
                state.close()
            app.state.coordinator_service.close()


@pytest.mark.asyncio
async def test_worker_death_reassignment_rejects_late_old_result_over_http(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_worker(registration(worker_id="worker-1", session_id="session-1"))
        repository.register_worker(registration(worker_id="worker-2", session_id="session-2"))
        assignment = assignments(base_policy())[0]
        repository.create_group([assignment], max_attempts=2)
        old = repository.create_lease(
            assignment.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="lease-old",
            duration_seconds=1,
        )
        clock.now = 12
        repository.expire_leases()
        clock.now = 14
        new = repository.create_lease(
            assignment.assignment_id,
            worker_id="worker-2",
            worker_session_id="session-2",
            lease_id="lease-new",
            duration_seconds=10,
        )
        app = create_coordinator_app(repository, token=TOKEN, trainer_ready=lambda: True)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                late = await client.put(
                    f"/api/v1/assignments/{assignment.assignment_id}/result",
                    headers=MSGPACK_HEADERS,
                    content=result_envelope_bytes(result_envelope(old, 1.0)),
                )
                assert late.status_code == 409
                accepted = await client.put(
                    f"/api/v1/assignments/{assignment.assignment_id}/result",
                    headers=MSGPACK_HEADERS,
                    content=result_envelope_bytes(result_envelope(new, 1.0)),
                )
                assert accepted.status_code == 200
                assert accepted.json()["terminal"] is True
                assert repository.assignment_state(assignment.assignment_id) == "succeeded"
        finally:
            app.state.coordinator_service.close()


@pytest.mark.asyncio
async def test_policy_transition_prefers_active_policy_and_cancels_stale_inflight_work(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.configure_scheduler(max_policy_lag=0, loaded_policy_preference_seconds=5)
        worker_registration = registration(caps=capabilities(capacity=2))
        repository.register_worker(worker_registration)
        repository.register_scheduler_source(train_source(group_size=1))
        app = create_coordinator_app(repository, token=TOKEN, trainer_ready=lambda: True, lease_duration_seconds=10)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                old_request = LeaseRequest(
                    request_id="request-old",
                    worker_id="worker-1",
                    worker_session_id="session-1",
                    sent_at=clock.now,
                    environments=(EnvironmentIdentity(id="env", revision="1"),),
                    available_slots=1,
                )
                old_response = await client.post(
                    "/api/v1/assignments/lease", headers=JSON_HEADERS, content=canonical_json_bytes(old_request)
                )
                assert old_response.status_code == 200
                old_lease = old_response.json()
                assert old_lease["assignment"]["policy"]["policy_version"] == 0

                policy1 = publish_policy(repository, 1, 11)
                repository.activate_policy(policy1.policy_id)
                heartbeat = WorkerHeartbeat(
                    worker_id="worker-1",
                    worker_session_id="session-1",
                    sent_at=clock.now,
                    active_lease_ids=(old_lease["lease_id"],),
                )
                stopped = await client.post(
                    "/api/v1/workers/heartbeat", headers=JSON_HEADERS, content=canonical_json_bytes(heartbeat)
                )
                assert stopped.status_code == 200
                assert stopped.json()["stop_lease_ids"] == [old_lease["lease_id"]]

                new_request = old_request.model_copy(
                    update={
                        "request_id": "request-new",
                        "sent_at": clock.now + 1,
                        "loaded_policy_ids": (policy1.policy_id,),
                    }
                )
                new_response = await client.post(
                    "/api/v1/assignments/lease", headers=JSON_HEADERS, content=canonical_json_bytes(new_request)
                )
                assert new_response.status_code == 200
                assert new_response.json()["assignment"]["policy"]["policy_id"] == policy1.policy_id
        finally:
            app.state.coordinator_service.close()


@pytest.mark.asyncio
async def test_restarted_worker_uploads_recovered_spool_entry_over_http(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        worker_registration = registration(caps=capabilities(capacity=2))
        repository.register_worker(worker_registration)
        assignment = assignments(base_policy())[0]
        repository.create_group([assignment], max_attempts=1)
        lease = repository.create_lease(
            assignment.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="lease-spooled",
            duration_seconds=10,
        )
        state_root = tmp_path / "restarted-worker"
        with WorkerState(state_root) as state:
            pending = WorkerSpool(state).publish(result_envelope(lease, 1.0))

        app = create_coordinator_app(repository, token=TOKEN, trainer_ready=lambda: True)
        client = StopAfterSubmitClient(
            "http://test",
            TOKEN,
            timeout_seconds=1,
            client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test"),
        )
        try:
            with WorkerState(state_root) as restarted:
                daemon = WorkerDaemon(
                    worker_config(tmp_path),
                    worker_registration,
                    client,
                    WorkerSpool(restarted),
                    RewardExecutor(0.0),
                    timestamp_sequence=TimestampSequence(lambda: clock.now),
                    request_id_factory=lambda: "request-no-work",
                )
                client.daemon = daemon
                await asyncio.wait_for(daemon.run(), timeout=5)
                assert repository.connection.execute("SELECT COUNT(*) FROM accepted_results").fetchone()[0] == 1
                assert not pending.path.exists()
                assert WorkerSpool(restarted).entries() == ()
        finally:
            await client.close()
            app.state.coordinator_service.close()


@pytest.mark.asyncio
async def test_restart_processes_accepted_unprocessed_results(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_worker(registration(caps=capabilities(capacity=2)))
        repository.register_scheduler_source(train_source())
        group = repository.create_next_group("train")
        leases = [
            repository.create_lease(
                assignment.assignment_id,
                worker_id="worker-1",
                worker_session_id="session-1",
                lease_id=f"lease-{index}",
                duration_seconds=10,
            )
            for index, assignment in enumerate(group.assignments)
        ]
        repository.accept_result(result_envelope(leases[0], 1.0))
        repository.accept_result(result_envelope(leases[1], 0.0))
        claimed = repository.claim_pending_results(1)
        assert len(claimed) == 1

    with make_repository(tmp_path, clock) as reopened:
        assert (
            reopened.connection.execute(
                "SELECT COUNT(*) FROM accepted_results WHERE processing_state = 'pending'"
            ).fetchone()[0]
            == 2
        )
        processor = RemoteResultProcessor(
            reopened,
            (
                ResultProcessingSource(
                    source_id="train-source",
                    environment=EnvironmentIdentity(id="env", revision="1"),
                    processing_id="grpo-v1",
                    algorithm=GRPOAlgorithm(GRPOAlgoConfig(), None),  # type: ignore[arg-type]
                ),
            ),
            batch_size=2,
        )
        assert await processor.process_available() == (1, 1)
        assert len(reopened.training_batches()) == 1
