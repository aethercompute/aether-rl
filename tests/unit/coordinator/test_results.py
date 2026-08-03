from pathlib import Path

import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, SamplingConfig, UserMessage

from aether_rl.configs.algorithm import GRPOAlgoConfig
from aether_rl.coordinator import (
    ArtifactCorruptionError,
    ConflictError,
    CoordinatorTrainingBatchExporter,
    ResultProcessingSource,
    ResultProcessor,
    decode_training_batch,
)
from aether_rl.coordinator.environments import EnvironmentSourceSpec
from aether_rl.orchestrator.algo import GRPOAlgorithm
from aether_rl.protocol import EnvironmentIdentity, ResultEnvelope, episode_digest, policy_manifest_digest
from aether_rl.transport.filesystem import BATCH_FILE_NAME
from aether_rl.utils.pathing import get_rollout_dir, get_step_path
from tests.unit.coordinator.test_database import FakeClock, capabilities, failure_envelope, registration
from tests.unit.coordinator.test_scheduler import make_repository, publish_policy


def completed_episode(assignment, reward: float) -> vf.WireEpisode:
    task = vf.TraceTask(type="Task", data=vf.WireTaskData.model_validate(assignment.task_data))
    trace = vf.WireTrace(
        task=task,
        nodes=[
            MessageNode(
                message=UserMessage(role="user", content="prompt"),
                token_ids=[1],
                mask=[False],
                routed_experts=np.zeros((1, 1, 1), dtype=np.uint8),
            ),
            MessageNode(
                parent=0,
                message=AssistantMessage(role="assistant", content="answer"),
                sampled=True,
                token_ids=[2, 3],
                mask=[False, True],
                logprobs=[-0.25],
                routed_experts=np.ones((2, 1, 1), dtype=np.uint8),
            ),
        ],
        rewards={"reward": reward},
        is_completed=True,
        ok=True,
    )
    return vf.WireEpisode(id=assignment.assignment_id, env=assignment.environment.id, ok=True, traces=[trace])


def result_envelope(lease, reward: float) -> ResultEnvelope:
    episode = completed_episode(lease.assignment, reward)
    policy_digest = policy_manifest_digest(lease.assignment.policy)
    return ResultEnvelope(
        assignment_id=lease.assignment.assignment_id,
        attempt=lease.attempt,
        lease_id=lease.lease_id,
        worker_id=lease.worker_id,
        worker_session_id=lease.worker_session_id,
        requested_policy_id=lease.assignment.policy.policy_id,
        served_policy_id=lease.assignment.policy.policy_id,
        requested_policy_digest=policy_digest,
        served_policy_digest=policy_digest,
        completed_at=11,
        result_digest=episode_digest(episode),
        episode=episode,
    )


@pytest.mark.asyncio
async def test_remote_results_finalize_grpo_emit_batch_and_replay_idempotently(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_worker(registration(caps=capabilities(capacity=2)))
        spec = EnvironmentSourceSpec(
            source_id="train-source",
            kind="train",
            environment=EnvironmentIdentity(id="env", revision="1"),
            tasks=({"idx": 0, "prompt": "prompt"},),
            sampling=SamplingConfig(temperature=1, max_tokens=8),
            group_size=2,
            max_attempts=1,
        )
        repository.register_scheduler_source(spec)
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
        repository.accept_result(result_envelope(leases[1], 0))
        repository.accept_result(result_envelope(leases[0], 1))

        algorithm = GRPOAlgorithm(GRPOAlgoConfig(), None)  # type: ignore[arg-type]
        processor = ResultProcessor(
            repository,
            (
                ResultProcessingSource(
                    source_id=spec.source_id,
                    environment=spec.environment,
                    processing_id="grpo-v1",
                    algorithm=algorithm,
                ),
            ),
            batch_size=2,
        )
        assert await processor.process_available() == (1, 1)
        records = repository.training_batches()
        assert len(records) == 1
        batch = decode_training_batch(records[0])
        assert batch.step == 1
        assert len(batch.examples) == 2
        assert [sample.advantages for sample in batch.examples] == [
            [0.0, 0.0, 0.5],
            [0.0, 0.0, -0.5],
        ]
        assert all(sample.temperatures == [1, 1, 1] for sample in batch.examples)
        assert all(sample.behavior_policy_id == group.assignments[0].policy.policy_id for sample in batch.examples)
        assert all(sample.behavior_policy_version == 0 for sample in batch.examples)
        assert all(sample.behavior_policy_digest is not None for sample in batch.examples)
        assert all(sample.routed_experts is not None for sample in batch.examples)
        metrics = processor.metrics.snapshot()
        assert metrics["inference/agg/informative_group_fraction"] == 1
        assert metrics["inference/agg/policy_lag"] == 0
        assert await processor.process_available() == (0, 0)
        assert repository.training_batches() == records

        exporter = CoordinatorTrainingBatchExporter(
            repository,
            tmp_path / "trainer",
            run_id="run_distributed",
            run_config=b"output_dir = 'unused'\n",
        )
        assert exporter.export_available() == 1
        assert (
            tmp_path / "trainer" / "run_distributed" / "control" / "orch.toml"
        ).read_bytes() == b"output_dir = 'unused'\n"
        exported_path = get_step_path(get_rollout_dir(tmp_path / "trainer" / "run_distributed"), 1) / BATCH_FILE_NAME
        assert exported_path.read_bytes() == records[0].artifact_path.read_bytes()
        assert exporter.export_available() == 0

        eval_spec = spec.model_copy(
            update={
                "source_id": "eval-source",
                "kind": "eval",
                "group_size": 1,
                "sampling": SamplingConfig(temperature=0, max_tokens=8),
            }
        )
        repository.register_scheduler_source(eval_spec)
        eval_group = repository.create_next_group("eval")
        eval_lease = repository.create_lease(
            eval_group.assignments[0].assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="eval-lease",
            duration_seconds=10,
        )
        repository.accept_result(result_envelope(eval_lease, 1))
        processor = ResultProcessor(
            repository,
            (
                ResultProcessingSource(
                    source_id=spec.source_id,
                    environment=spec.environment,
                    processing_id="grpo-v1",
                    algorithm=algorithm,
                ),
                ResultProcessingSource(
                    source_id=eval_spec.source_id,
                    environment=eval_spec.environment,
                    processing_id="eval-v1",
                ),
            ),
            batch_size=2,
        )
        assert await processor.process_available() == (1, 0)
        assert len(repository.processed_groups()) == 2
        assert repository.training_batches() == records

        records[0].artifact_path.write_bytes(b"corrupt")
        with pytest.raises(ArtifactCorruptionError, match="digest|size"):
            ResultProcessor(repository, (), batch_size=1)


@pytest.mark.asyncio
async def test_training_batch_export_rejects_conflicting_trainer_file(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path / "coordinator", clock) as repository:
        repository.register_worker(registration(caps=capabilities(capacity=2)))
        spec = EnvironmentSourceSpec(
            source_id="train-source",
            kind="train",
            environment=EnvironmentIdentity(id="env", revision="1"),
            tasks=({"idx": 0, "prompt": "prompt"},),
            sampling=SamplingConfig(temperature=1, max_tokens=8),
            group_size=2,
            max_attempts=1,
        )
        repository.register_scheduler_source(spec)
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
        repository.accept_result(result_envelope(leases[0], 0))
        repository.accept_result(result_envelope(leases[1], 1))
        processor = ResultProcessor(
            repository,
            (
                ResultProcessingSource(
                    source_id=spec.source_id,
                    environment=spec.environment,
                    processing_id="grpo-v1",
                    algorithm=GRPOAlgorithm(GRPOAlgoConfig(), None),  # type: ignore[arg-type]
                ),
            ),
            batch_size=2,
        )
        assert await processor.process_available() == (1, 1)
        exported_path = get_step_path(get_rollout_dir(tmp_path / "trainer" / "run_distributed"), 1) / BATCH_FILE_NAME
        exported_path.parent.mkdir(parents=True)
        exported_path.write_bytes(b"different")
        exporter = CoordinatorTrainingBatchExporter(
            repository,
            tmp_path / "trainer",
            run_id="run_distributed",
            run_config=b"output_dir = 'unused'\n",
        )
        with pytest.raises(ArtifactCorruptionError, match="conflicts"):
            exporter.export_available()


@pytest.mark.asyncio
async def test_emit_batches_partition_processed_rollouts_by_policy_identity(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.configure_scheduler(max_policy_lag=100, loaded_policy_preference_seconds=0)
        repository.register_worker(registration(caps=capabilities(capacity=4)))
        spec = EnvironmentSourceSpec(
            source_id="train-source",
            kind="train",
            environment=EnvironmentIdentity(id="env", revision="1"),
            tasks=({"idx": 0, "prompt": "prompt"},),
            sampling=SamplingConfig(temperature=1, max_tokens=8),
            group_size=1,
            max_attempts=1,
        )
        repository.register_scheduler_source(spec)

        for index, version in enumerate((10, 11, 12, 12)):
            if repository.active_policy().policy_version != version:
                policy = publish_policy(repository, version, created_at=20.0 + version)
                repository.activate_policy(policy.policy_id)
            group = repository.create_next_group("train")
            lease = repository.create_lease(
                group.assignments[0].assignment_id,
                worker_id="worker-1",
                worker_session_id="session-1",
                lease_id=f"policy-{version}-lease-{index}",
                duration_seconds=10,
            )
            repository.accept_result(result_envelope(lease, 1.0))

        processor = ResultProcessor(
            repository,
            (
                ResultProcessingSource(
                    source_id=spec.source_id,
                    environment=spec.environment,
                    processing_id="grpo-v1",
                    algorithm=GRPOAlgorithm(GRPOAlgoConfig(), None),  # type: ignore[arg-type]
                ),
            ),
            batch_size=2,
        )

        assert await processor.process_available() == (4, 1)
        batch = decode_training_batch(repository.training_batches()[0])
        assert {sample.behavior_policy_version for sample in batch.examples} == {12}
        assert len({sample.behavior_policy_digest for sample in batch.examples}) == 1
        assert processor.metrics.mixed_policy_batch_attempts == 2


@pytest.mark.asyncio
async def test_emit_batches_drop_stale_processed_rollouts(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.configure_scheduler(max_policy_lag=0, loaded_policy_preference_seconds=0)
        repository.register_worker(registration(caps=capabilities(capacity=1)))
        spec = EnvironmentSourceSpec(
            source_id="train-source",
            kind="train",
            environment=EnvironmentIdentity(id="env", revision="1"),
            tasks=({"idx": 0, "prompt": "prompt"},),
            sampling=SamplingConfig(temperature=1, max_tokens=8),
            group_size=1,
            max_attempts=1,
        )
        repository.register_scheduler_source(spec)

        policy_v10 = publish_policy(repository, 10, created_at=30.0)
        repository.activate_policy(policy_v10.policy_id)
        group = repository.create_next_group("train")
        lease = repository.create_lease(
            group.assignments[0].assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="policy-10-lease",
            duration_seconds=10,
        )
        repository.accept_result(result_envelope(lease, 1.0))

        processor = ResultProcessor(
            repository,
            (
                ResultProcessingSource(
                    source_id=spec.source_id,
                    environment=spec.environment,
                    processing_id="grpo-v1",
                    algorithm=GRPOAlgorithm(GRPOAlgoConfig(), None),  # type: ignore[arg-type]
                ),
            ),
            batch_size=2,
        )
        assert await processor.process_available() == (1, 0)

        policy_v11 = publish_policy(repository, 11, created_at=31.0)
        repository.activate_policy(policy_v11.policy_id)

        assert await processor.process_available() == (0, 0)
        assert processor.metrics.stale_processed_rollouts_dropped == 1
        assert processor.metrics.snapshot()["inference/agg/stale_drops"] == 1
        assert repository.training_batches() == ()
        assert repository.pending_processed_rollouts() == ()


@pytest.mark.asyncio
async def test_group_scored_source_drops_partial_terminal_group(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_worker(registration(caps=capabilities(capacity=2)))
        spec = EnvironmentSourceSpec(
            source_id="group-scored",
            kind="train",
            environment=EnvironmentIdentity(id="env", revision="1"),
            tasks=({"idx": 0, "prompt": "prompt"},),
            sampling=SamplingConfig(temperature=1, max_tokens=8),
            group_size=2,
            max_attempts=1,
        )
        repository.register_scheduler_source(spec)
        group = repository.create_next_group("train")
        leases = [
            repository.create_lease(
                assignment.assignment_id,
                worker_id="worker-1",
                worker_session_id="session-1",
                lease_id=f"group-scored-lease-{index}",
                duration_seconds=10,
            )
            for index, assignment in enumerate(group.assignments)
        ]
        repository.accept_result(result_envelope(leases[0], 1))
        repository.accept_failure(failure_envelope(leases[1], retryable=False))
        processor = ResultProcessor(
            repository,
            (
                ResultProcessingSource(
                    source_id=spec.source_id,
                    environment=spec.environment,
                    processing_id="group-scored-v1",
                    algorithm=GRPOAlgorithm(GRPOAlgoConfig(), None),  # type: ignore[arg-type]
                    requires_group_scoring=True,
                ),
            ),
            batch_size=1,
        )
        assert await processor.process_available() == (1, 0)
        assert (
            repository.connection.execute(
                "SELECT rollout_count FROM processed_groups WHERE group_id = ?", (group.group_id,)
            ).fetchone()[0]
            == 0
        )


@pytest.mark.asyncio
async def test_out_of_order_policy_soak_retries_duplicates_and_stale_drops(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.configure_scheduler(max_policy_lag=100, loaded_policy_preference_seconds=0)
        repository.register_worker(registration(caps=capabilities(capacity=64)))
        spec = EnvironmentSourceSpec(
            source_id="train-source",
            kind="train",
            environment=EnvironmentIdentity(id="env", revision="1"),
            tasks=({"idx": 0, "prompt": "prompt"},),
            sampling=SamplingConfig(temperature=1, max_tokens=8),
            group_size=1,
            max_attempts=2,
        )
        repository.register_scheduler_source(spec)

        policies = {
            version: publish_policy(repository, version, created_at=100.0 + version) for version in range(10, 14)
        }
        created = []
        valid_envelopes = []
        old_envelopes = []
        sequence = [10] * 8 + [11] * 8 + [12] * 8
        for index, version in enumerate(sequence):
            if repository.active_policy().policy_version != version:
                repository.activate_policy(policies[version].policy_id)
            group = repository.create_next_group("train")
            assignment = group.assignments[0]
            lease = repository.create_lease(
                assignment.assignment_id,
                worker_id="worker-1",
                worker_session_id="session-1",
                lease_id=f"lease-{index}-a",
                duration_seconds=1 if index % 5 == 0 else 1000,
            )
            if index % 5 == 0:
                old_envelopes.append(result_envelope(lease, float(index % 2)))
                clock.now += 2
                repository.expire_leases()
                assert repository.assignment_state(assignment.assignment_id) == "retry_wait"
                clock.now += 2
                retry = repository.create_lease(
                    assignment.assignment_id,
                    worker_id="worker-1",
                    worker_session_id="session-1",
                    lease_id=f"lease-{index}-b",
                    duration_seconds=1000,
                )
                valid_envelopes.append(result_envelope(retry, float(index % 2)))
            else:
                valid_envelopes.append(result_envelope(lease, float(index % 2)))
            created.append((assignment.assignment_id, version))

        for index, envelope in reversed(list(enumerate(valid_envelopes))):
            accepted = repository.accept_result(envelope)
            assert accepted.terminal and not accepted.duplicate
            if index % 3 == 0:
                duplicate = repository.accept_result(envelope)
                assert duplicate.terminal and duplicate.duplicate
        for envelope in old_envelopes:
            with pytest.raises(ConflictError):
                repository.accept_result(envelope)

        processor = ResultProcessor(
            repository,
            (
                ResultProcessingSource(
                    source_id=spec.source_id,
                    environment=spec.environment,
                    processing_id="grpo-v1",
                    algorithm=GRPOAlgorithm(GRPOAlgoConfig(), None),  # type: ignore[arg-type]
                ),
            ),
            batch_size=8,
        )
        assert await processor.process_available() == (24, 3)
        batches = [decode_training_batch(record) for record in repository.training_batches()]
        assert len(batches) == 3
        assert sorted({sample.behavior_policy_version for sample in batch.examples}.pop() for batch in batches) == [
            10,
            11,
            12,
        ]
        for batch in batches:
            assert len(batch.examples) == 8
            assert len({sample.behavior_policy_version for sample in batch.examples}) == 1
            assert len({sample.behavior_policy_digest for sample in batch.examples}) == 1
        assert len(created) == 24

        repository.configure_scheduler(max_policy_lag=100, loaded_policy_preference_seconds=0)
        repository.activate_policy(policies[13].policy_id)
        stale_assignments = []
        for index in range(3):
            group = repository.create_next_group("train")
            assignment = group.assignments[0]
            stale_assignments.append(assignment.assignment_id)
            lease = repository.create_lease(
                assignment.assignment_id,
                worker_id="worker-1",
                worker_session_id="session-1",
                lease_id=f"stale-lease-{index}",
                duration_seconds=20,
            )
            repository.accept_result(result_envelope(lease, 1.0))

        stale_processor = ResultProcessor(
            repository,
            (
                ResultProcessingSource(
                    source_id=spec.source_id,
                    environment=spec.environment,
                    processing_id="grpo-v1",
                    algorithm=GRPOAlgorithm(GRPOAlgoConfig(), None),  # type: ignore[arg-type]
                ),
            ),
            batch_size=8,
        )
        assert await stale_processor.process_available() == (3, 0)
        newer = publish_policy(repository, 14, created_at=114.0)
        repository.record_policy(newer, repository.run_root / "policies" / newer.policy_id)
        repository.activate_policy(newer.policy_id)
        repository.configure_scheduler(max_policy_lag=0, loaded_policy_preference_seconds=0)
        assert await stale_processor.process_available() == (0, 0)
        assert stale_processor.metrics.stale_processed_rollouts_dropped == 3
        assert all(repository.assignment_state(assignment_id) == "succeeded" for assignment_id in stale_assignments)
