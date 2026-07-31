from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
import verifiers.v1 as vf

from aether_rl.configs.worker import WorkerConfig
from aether_rl.protocol import AssignmentLease
from aether_rl.worker import executor as worker_executor
from aether_rl.worker.executor import VerifiersAssignmentExecutor
from tests.unit.coordinator.test_database import assignments, base_model, base_policy


class FakeClient:
    def __init__(self):
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class FakeTaskData(vf.TaskData):
    answer: str


class FakeTask(vf.Task[FakeTaskData]):
    pass


class FakeTaskset:
    def __init__(self):
        self.config = type("Config", (), {"task": vf.TaskConfig()})()

    @classmethod
    def task_type(cls):
        return FakeTask


class FakeEnvConfig:
    env_id = "env"
    id = ""
    max_concurrent = 2
    taskset = type("TasksetConfig", (), {"id": "env"})()


class FakeEnv:
    def __init__(self, episode: vf.WireEpisode):
        self.config = FakeEnvConfig()
        self.taskset = FakeTaskset()
        self.episode = episode
        self.model = None
        self.sampling = None
        self.gate = None
        self.entered = False

    @asynccontextmanager
    async def serving(self):
        self.entered = True
        yield

    def slots(self, task):
        assert task.data.answer == "ok"
        return [object()]

    async def run_slot(self, slot, context, gate=None):
        self.model = context.model
        self.sampling = context.sampling
        self.gate = gate
        return self.episode


def worker_config(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "coordinator_url": "https://coordinator.example.com",
            "state_dir": tmp_path / "worker",
            "base_model": base_model().model_dump(mode="python"),
            "environments": [
                {
                    "id": "env",
                    "package": "env",
                    "revision": "1",
                    "config": {"taskset": {"id": "env"}},
                }
            ],
        }
    )


def lease() -> AssignmentLease:
    assignment = assignments(base_policy())[0].model_copy(update={"task_data": {"answer": "ok"}})
    return AssignmentLease(
        lease_id="lease-1",
        attempt=1,
        worker_id="worker-1",
        worker_session_id="session-1",
        issued_at=3,
        expires_at=100,
        assignment=assignment,
    )


@pytest.mark.asyncio
async def test_executor_runs_verifiers_episode_against_loopback_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_client = FakeClient()
    fake_env = FakeEnv(vf.WireEpisode(env="env", ok=True, traces=[]))
    captured_client_config = None

    def resolve_client(config):
        nonlocal captured_client_config
        captured_client_config = config
        return fake_client

    monkeypatch.setattr(worker_executor, "resolve_env_config", lambda data: FakeEnvConfig())
    monkeypatch.setattr(worker_executor, "load_environment", lambda config: fake_env)
    monkeypatch.setattr(worker_executor, "resolve_client", resolve_client)

    executor = VerifiersAssignmentExecutor(worker_config(tmp_path))
    result = await executor.execute(lease(), cancel_event=worker_executor.asyncio.Event())

    assert result.type == "result"
    assert result.requested_policy_id == lease().assignment.policy.policy_id
    assert result.served_policy_id == lease().assignment.policy.policy_id
    assert fake_env.entered
    assert fake_env.model == lease().assignment.policy.served_model_name
    assert fake_env.gate is not None
    assert captured_client_config.base_url == "http://127.0.0.1:8000/v1"
    assert captured_client_config.renderer_model_name == base_model().model_name
    assert captured_client_config.renderer_model_revision == base_model().tokenizer_revision
    assert fake_client.closed


def test_executor_rejects_catalog_identity_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class WrongEnvConfig:
        env_id = "wrong"
        id = ""
        taskset = type("TasksetConfig", (), {"id": "env"})()

    monkeypatch.setattr(worker_executor, "resolve_env_config", lambda data: WrongEnvConfig())
    with pytest.raises(RuntimeError, match="resolves to 'wrong'"):
        VerifiersAssignmentExecutor(worker_config(tmp_path))


def test_executor_rejects_package_that_does_not_match_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(worker_executor, "resolve_env_config", lambda data: FakeEnvConfig())
    config = worker_config(tmp_path).model_copy(
        update={"environments": [worker_config(tmp_path).environments[0].model_copy(update={"package": "other"})]}
    )
    with pytest.raises(RuntimeError, match="package must match"):
        VerifiersAssignmentExecutor(config)


@pytest.mark.asyncio
async def test_executor_enforces_result_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(worker_executor, "resolve_env_config", lambda data: FakeEnvConfig())
    monkeypatch.setattr(
        worker_executor, "load_environment", lambda config: FakeEnv(vf.WireEpisode(env="env", ok=True, traces=[]))
    )
    monkeypatch.setattr(worker_executor, "resolve_client", lambda config: FakeClient())

    assignment = lease().assignment.model_copy(update={"result_size_limit_bytes": 1})
    limited = lease().model_copy(update={"assignment": assignment})
    executor = VerifiersAssignmentExecutor(worker_config(tmp_path))
    with pytest.raises(worker_executor.TerminalExecutionError, match="result exceeds") as error:
        await executor.execute(limited, cancel_event=worker_executor.asyncio.Event())
    assert error.value.retryable is False
    assert error.value.code == "result_too_large"


@pytest.mark.asyncio
async def test_executor_cancels_during_environment_startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    fake_client = FakeClient()
    serving_started = worker_executor.asyncio.Event()
    release_serving = worker_executor.asyncio.Event()

    class HangingEnv(FakeEnv):
        @asynccontextmanager
        async def serving(self):
            serving_started.set()
            await release_serving.wait()
            yield

    monkeypatch.setattr(worker_executor, "resolve_env_config", lambda data: FakeEnvConfig())
    monkeypatch.setattr(worker_executor, "load_environment", lambda config: HangingEnv(vf.WireEpisode(env="env")))
    monkeypatch.setattr(worker_executor, "resolve_client", lambda config: fake_client)

    cancel_event = worker_executor.asyncio.Event()
    executor = VerifiersAssignmentExecutor(worker_config(tmp_path))
    running = worker_executor.asyncio.create_task(executor.execute(lease(), cancel_event=cancel_event))
    await serving_started.wait()
    cancel_event.set()
    with pytest.raises(worker_executor.asyncio.CancelledError):
        await running
    assert fake_client.closed
