import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors.torch import save_file

import aether_rl.coordinator.runtime as runtime_module
from aether_rl.coordinator.runtime import CoordinatorRuntime, prune_published_trainer_artifacts
from tests.unit.coordinator.test_database import base_policy
from tests.unit.train.test_policy import adapter_state_dict


def test_load_tasks_shuffles_finite_taskset_before_limit(monkeypatch):
    class FiniteTaskset:
        INFINITE = False

        def load(self):
            return range(10)

    environment = SimpleNamespace(id="env", taskset=object())
    monkeypatch.setattr(runtime_module, "resolve_env_config", lambda _: environment)
    monkeypatch.setattr(runtime_module.vf, "load_taskset", lambda _: FiniteTaskset())
    source = SimpleNamespace(
        source_id="source",
        environment_id="env",
        environment_config={},
        task_limit=4,
        shuffle_seed=42,
    )

    first = CoordinatorRuntime._load_tasks(source)
    second = CoordinatorRuntime._load_tasks(source)

    assert first == second
    assert len(first) == 4
    assert sorted(first) != [0, 1, 2, 3]


def test_load_tasks_rejects_shuffle_for_infinite_taskset(monkeypatch):
    class InfiniteTaskset:
        INFINITE = True

        def load(self):
            return iter(range(10))

    environment = SimpleNamespace(id="env", taskset=object())
    monkeypatch.setattr(runtime_module, "resolve_env_config", lambda _: environment)
    monkeypatch.setattr(runtime_module.vf, "load_taskset", lambda _: InfiniteTaskset())
    source = SimpleNamespace(
        source_id="source",
        environment_id="env",
        environment_config={},
        task_limit=4,
        shuffle_seed=42,
    )

    with pytest.raises(ValueError, match="cannot shuffle an infinite taskset"):
        CoordinatorRuntime._load_tasks(source)


def make_step_dirs(root: Path, versions: range) -> None:
    for version in versions:
        (root / f"step_{version}").mkdir(parents=True)


def test_prune_published_trainer_artifacts_preserves_recent_and_future(tmp_path: Path):
    checkpoints = tmp_path / "checkpoints"
    broadcasts = tmp_path / "run_run-1" / "broadcasts"
    make_step_dirs(checkpoints, range(1, 7))
    make_step_dirs(broadcasts, range(1, 7))
    (checkpoints / "step_incomplete").mkdir()

    prune_published_trainer_artifacts(
        tmp_path,
        "run_run-1",
        active_version=5,
        checkpoint_keep_last=2,
    )

    assert {path.name for path in checkpoints.iterdir()} == {
        "step_4",
        "step_5",
        "step_6",
        "step_incomplete",
    }
    assert {path.name for path in broadcasts.iterdir()} == {"step_6"}


def test_prune_published_trainer_artifacts_continues_after_delete_error(tmp_path: Path, monkeypatch, caplog):
    checkpoints = tmp_path / "checkpoints"
    broadcasts = tmp_path / "run_run-1" / "broadcasts"
    make_step_dirs(checkpoints, range(1, 4))
    make_step_dirs(broadcasts, range(1, 4))
    original_rmtree = runtime_module.shutil.rmtree

    def fail_first_checkpoint(path, *, dir_fd=None):
        if path == "step_1" and dir_fd is not None:
            opened = os.fstat(dir_fd)
            expected = checkpoints.stat()
            if (opened.st_dev, opened.st_ino) == (expected.st_dev, expected.st_ino):
                raise OSError("permission denied")
        original_rmtree(path, dir_fd=dir_fd)

    monkeypatch.setattr(runtime_module.shutil, "rmtree", fail_first_checkpoint)
    prune_published_trainer_artifacts(
        tmp_path,
        "run_run-1",
        active_version=3,
        checkpoint_keep_last=1,
    )

    assert (checkpoints / "step_1").is_dir()
    assert not (checkpoints / "step_2").exists()
    assert (checkpoints / "step_3").is_dir()
    assert not any(broadcasts.iterdir())
    assert "permission denied" in caplog.text


def test_prune_published_trainer_artifacts_rejects_symlinked_root(tmp_path: Path, caplog):
    outside_checkpoints = tmp_path / "outside-checkpoints"
    outside_run = tmp_path / "outside-run"
    outside_broadcasts = outside_run / "broadcasts"
    make_step_dirs(outside_checkpoints, range(1, 3))
    make_step_dirs(outside_broadcasts, range(1, 3))
    (tmp_path / "checkpoints").symlink_to(outside_checkpoints, target_is_directory=True)
    (tmp_path / "run_run-1").symlink_to(outside_run, target_is_directory=True)

    prune_published_trainer_artifacts(
        tmp_path,
        "run_run-1",
        active_version=2,
        checkpoint_keep_last=1,
    )

    assert {path.name for path in outside_checkpoints.iterdir()} == {"step_1", "step_2"}
    assert {path.name for path in outside_broadcasts.iterdir()} == {"step_1", "step_2"}
    assert "failed to inspect published trainer artifacts" in caplog.text


def test_policy_activation_precedes_artifact_pruning(tmp_path: Path, monkeypatch):
    policy = base_policy()

    class Repository:
        active = policy

        def active_policy(self):
            return self.active

        def record_and_activate_policy(self, manifest, artifact_path):
            assert artifact_path.is_dir()
            self.active = manifest
            return manifest

    class Service:
        async def call(self, function, *args):
            return function(*args)

    repository = Repository()
    runtime = object.__new__(CoordinatorRuntime)
    runtime.config = SimpleNamespace(
        run_root=tmp_path / "run",
        run_id="run-1",
        published_checkpoint_keep_last=2,
    )
    runtime.repository = repository
    runtime.base_policy = policy
    runtime.service = Service()
    runtime.trainer_output_dir = tmp_path / "trainer"
    runtime.trainer_run_id = "run_run-1"
    runtime.trainer_config = SimpleNamespace(model=SimpleNamespace(lora=SimpleNamespace(rank=2, alpha=4, dropout=0.0)))
    runtime.config.run_root.mkdir()

    broadcast = runtime.trainer_output_dir / runtime.trainer_run_id / "broadcasts" / "step_1"
    checkpoint = runtime.trainer_output_dir / "checkpoints" / "step_1"
    broadcast.mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    save_file(adapter_state_dict(), broadcast / "adapter_model.safetensors")
    (broadcast / "STABLE").touch()
    (checkpoint / "STABLE").touch()

    original_prune = runtime_module.prune_published_trainer_artifacts

    def assert_activated_before_pruning(*args, **kwargs):
        assert repository.active.policy_version == 1
        original_prune(*args, **kwargs)

    monkeypatch.setattr(runtime_module, "prune_published_trainer_artifacts", assert_activated_before_pruning)

    assert asyncio.run(runtime._publish_available_policies())
    assert repository.active.policy_version == 1
    assert checkpoint.is_dir()
    assert not broadcast.exists()
