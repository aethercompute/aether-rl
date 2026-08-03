import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from aether_rl.coordinator import environments as environment_module
from aether_rl.coordinator.environments import CentralEpisodeRunner
from aether_rl.coordinator.inference import BrokerTransport, InferenceBroker
from aether_rl.protocol import InferenceReply, decode_result_envelope
from tests.unit.coordinator.test_database import FakeClock, assignments, base_policy, registration
from tests.unit.coordinator.test_results import completed_episode
from tests.unit.coordinator.test_scheduler import make_repository


class FakeData:
    @classmethod
    def model_validate(cls, value):
        return value


class FakeTask:
    def __init__(self, data, config):
        self.data = data


class FakeTaskset:
    config = SimpleNamespace(task=None)

    @classmethod
    def task_type(cls):
        return FakeTask


class FakeEnvironment:
    config = SimpleNamespace(max_concurrent=None)
    taskset = FakeTaskset()

    def __init__(self, assignment):
        self.assignment = assignment

    @asynccontextmanager
    async def serving(self):
        yield

    def slots(self, task):
        return (task,)

    async def run_slot(self, slot, context, gate):
        response = await context.client.generate()
        assert response.json() == {"generated": True}
        return completed_episode(self.assignment, 1.0)


class FakeTrainClient:
    def __init__(self, broker, lease_id):
        self.http = httpx.AsyncClient(
            transport=BrokerTransport(broker, lease_id),
            base_url="http://inference.invalid",
        )

    async def generate(self):
        return await self.http.post("/inference/v1/generate", json={"prompt_token_ids": [1, 2]})

    async def close(self):
        await self.http.aclose()


@pytest.mark.asyncio
async def test_central_environment_uses_worker_inference_and_spools_scored_result(tmp_path: Path, monkeypatch):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_worker(registration())
        assignment = assignments(base_policy())[0]
        repository.create_group([assignment], max_attempts=1)
        lease = repository.create_lease(
            assignment.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="lease-1",
            duration_seconds=100,
        )
        broker = InferenceBroker(body_limit_bytes=1024 * 1024)

        monkeypatch.setattr(environment_module, "load_environment", lambda config: FakeEnvironment(assignment))
        monkeypatch.setattr(environment_module, "task_data_cls", lambda task_type: FakeData)
        monkeypatch.setattr(
            environment_module,
            "create_train_client",
            lambda broker, lease_id, **kwargs: FakeTrainClient(broker, lease_id),
        )

        async def database_call(function, *args):
            return function(*args)

        runner = CentralEpisodeRunner(
            {assignment.source_id: object()},  # type: ignore[dict-item]
            broker,
            repository,
            database_call,
            renderer_model_name="org/model",
            renderer_model_revision="a" * 40,
            slots=1,
        )
        runner.start(lease)
        episode_task = runner.tasks[lease.lease_id]
        request = await broker.exchange("lease-1", "worker-1", "session-1", None, 1)
        assert request is not None and request is not False
        assert request.path == "/inference/v1/generate"
        assert b"prompt_token_ids" in request.body
        await broker.exchange(
            "lease-1",
            "worker-1",
            "session-1",
            InferenceReply(
                request_id=request.request_id,
                status_code=200,
                headers={"content-type": "application/json"},
                body=b'{"generated":true}',
            ),
            0,
        )
        await episode_task

        pending = repository.claim_pending_results(1)
        assert len(pending) == 1
        envelope = decode_result_envelope(pending[0].path.read_bytes())
        assert envelope.episode.traces[0].rewards == {"reward": 1.0}
        await runner.stop()


@pytest.mark.asyncio
async def test_inference_reply_submission_is_idempotent():
    broker = InferenceBroker(body_limit_bytes=1024)
    broker.register("lease-1", "worker-1", "session-1")
    response_task = asyncio.create_task(broker.request("lease-1", "GET", "/v1/models", {}, b""))
    request = await broker.exchange("lease-1", "worker-1", "session-1", None, 1)
    assert request is not None and request is not False
    reply = InferenceReply(
        request_id=request.request_id,
        status_code=200,
        headers={"content-type": "application/json"},
        body=b'{"data":[]}',
    )
    second_response = asyncio.create_task(broker.request("lease-1", "POST", "/inference/v1/generate", {}, b"{}"))
    second_request = await broker.exchange("lease-1", "worker-1", "session-1", reply, 1)
    assert second_request is not None and second_request is not False
    repeated = await broker.exchange("lease-1", "worker-1", "session-1", reply, 0)
    assert repeated == second_request
    assert await response_task == reply
    with pytest.raises(ValueError, match="different reply"):
        await broker.exchange(
            "lease-1",
            "worker-1",
            "session-1",
            reply.model_copy(update={"status_code": 500}),
            0,
        )
    second_reply = InferenceReply(
        request_id=second_request.request_id,
        status_code=200,
        body=b'{"choices":[]}',
    )
    await broker.exchange("lease-1", "worker-1", "session-1", second_reply, 0)
    assert await second_response == second_reply
    broker.close("lease-1")
