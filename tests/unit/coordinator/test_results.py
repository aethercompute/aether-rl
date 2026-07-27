from pathlib import Path

import numpy as np
import pytest
import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, SamplingConfig, UserMessage

from aether_rl.configs.algorithm import GRPOAlgoConfig
from aether_rl.coordinator import (
    ArtifactCorruptionError,
    RemoteResultProcessor,
    ResultProcessingSource,
    decode_training_batch,
)
from aether_rl.coordinator.environments import EnvironmentSourceSpec
from aether_rl.orchestrator.algo import GRPOAlgorithm
from aether_rl.protocol import ResultEnvelope, episode_digest, policy_manifest_digest
from tests.unit.coordinator.test_database import FakeClock, capabilities, failure_envelope, registration
from tests.unit.coordinator.test_scheduler import make_repository


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
            environment=registration().capabilities.environments[0],
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
        processor = RemoteResultProcessor(
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
        assert await processor.process_available() == (0, 0)
        assert repository.training_batches() == records

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
        processor = RemoteResultProcessor(
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
            RemoteResultProcessor(repository, (), batch_size=1)


@pytest.mark.asyncio
async def test_group_scored_source_drops_partial_terminal_group(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_worker(registration(caps=capabilities(capacity=2)))
        spec = EnvironmentSourceSpec(
            source_id="group-scored",
            kind="train",
            environment=registration().capabilities.environments[0],
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
        processor = RemoteResultProcessor(
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
