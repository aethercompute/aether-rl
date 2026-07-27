import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from verifiers.v1.episode import WireEpisode
from verifiers.v1.types import SamplingConfig

from aether_rl.coordinator import (
    ArtifactCorruptionError,
    AtomicSpool,
    CapacityError,
    ConflictError,
    CoordinatorLockError,
    CoordinatorRepository,
    IncompatibleWorkerError,
    InvalidStateError,
    SchemaVersionError,
)
from aether_rl.coordinator.migrations import MIGRATIONS
from aether_rl.protocol import (
    BaseModelIdentity,
    EnvironmentIdentity,
    FailureEnvelope,
    PolicyManifest,
    ResultEnvelope,
    RolloutAssignment,
    RuntimeIdentity,
    WorkerCapabilities,
    WorkerRegistration,
    canonical_json_bytes,
    episode_digest,
    policy_manifest_digest,
)
from aether_rl.trainer.policy import publish_lora_policy

DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
REVISION = "a" * 40


@dataclass
class FakeClock:
    now: float = 10.0

    def __call__(self) -> float:
        return self.now


def base_model() -> BaseModelIdentity:
    return BaseModelIdentity(
        model_name="org/model",
        model_revision=REVISION,
        model_config_digest=DIGEST_A,
        tokenizer_name="org/model",
        tokenizer_revision=REVISION,
        tokenizer_digest=DIGEST_B,
        chat_template_digest=DIGEST_A,
        vocab_size=128,
    )


def base_policy() -> PolicyManifest:
    return PolicyManifest(run_id="run-1", policy_version=0, base_model=base_model(), created_at=1.0)


def capabilities(*, capacity: int = 1, model: BaseModelIdentity | None = None) -> WorkerCapabilities:
    return WorkerCapabilities(
        base_model=model or base_model(),
        runtime=RuntimeIdentity(
            aether_rl_version="0.7.0",
            python_version="3.12",
            torch_version="2.11",
            transformers_version="5.6.2",
            vllm_version="0.24",
        ),
        environments=(EnvironmentIdentity(id="env", revision="1"),),
        max_concurrent_assignments=capacity,
        gpu_count=1,
        tensor_parallel_size=1,
    )


def registration(
    *, worker_id: str = "worker-1", session_id: str = "session-1", caps: WorkerCapabilities | None = None
) -> WorkerRegistration:
    return WorkerRegistration(
        worker_id=worker_id,
        worker_session_id=session_id,
        registered_at=2.0,
        capabilities=caps or capabilities(),
    )


def assignments(
    policy: PolicyManifest,
    *,
    group_id: str = "group-1",
    size: int = 1,
    deadline_at: float | None = 100.0,
) -> list[RolloutAssignment]:
    return [
        RolloutAssignment(
            assignment_id=f"{group_id}-assignment-{index}",
            group_id=group_id,
            group_index=index,
            group_size=size,
            kind="train",
            environment=EnvironmentIdentity(id="env", revision="1"),
            task_data={"prompt": "hello"},
            sampling=SamplingConfig(max_tokens=8),
            policy=policy,
            created_at=3.0,
            deadline_at=deadline_at,
        )
        for index in range(size)
    ]


def result_envelope(lease, *, completed_at: float = 11.0) -> ResultEnvelope:
    episode = WireEpisode(id=lease.assignment.assignment_id, env="env", ok=True)
    digest = policy_manifest_digest(lease.assignment.policy)
    return ResultEnvelope(
        assignment_id=lease.assignment.assignment_id,
        attempt=lease.attempt,
        lease_id=lease.lease_id,
        worker_id=lease.worker_id,
        worker_session_id=lease.worker_session_id,
        requested_policy_id=lease.assignment.policy.policy_id,
        served_policy_id=lease.assignment.policy.policy_id,
        requested_policy_digest=digest,
        served_policy_digest=digest,
        completed_at=completed_at,
        result_digest=episode_digest(episode),
        episode=episode,
    )


def failure_envelope(lease, *, retryable: bool = True, message: str = "failed") -> FailureEnvelope:
    return FailureEnvelope(
        assignment_id=lease.assignment.assignment_id,
        attempt=lease.attempt,
        lease_id=lease.lease_id,
        worker_id=lease.worker_id,
        worker_session_id=lease.worker_session_id,
        failed_at=11.0,
        code="environment-error",
        message=message,
        retryable=retryable,
    )


def repository(tmp_path: Path, clock: FakeClock) -> CoordinatorRepository:
    run_root = tmp_path / "run"
    return CoordinatorRepository(
        run_root / "control" / "coordinator.sqlite3",
        run_root,
        clock=clock,
        retry_base_seconds=2.0,
        retry_max_seconds=8.0,
    )


def initialized_repository(tmp_path: Path, clock: FakeClock) -> CoordinatorRepository:
    state = repository(tmp_path, clock)
    state.initialize_run(base_policy())
    return state


def test_pragmas_migrations_and_newer_schema_rejection(tmp_path: Path):
    clock = FakeClock()
    with repository(tmp_path, clock) as state:
        assert state.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert state.connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert state.connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert state.connection.execute("PRAGMA busy_timeout").fetchone()[0] == 5000
        assert state.connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 4
        assert not state.connection.execute("PRAGMA foreign_key_check").fetchall()
        with pytest.raises(CoordinatorLockError, match="run lock"):
            repository(tmp_path, clock)

    database_path = tmp_path / "newer.sqlite3"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY, applied_at REAL NOT NULL)")
    connection.execute("INSERT INTO schema_migrations VALUES (5, 0)")
    connection.commit()
    connection.close()
    with pytest.raises(SchemaVersionError, match="newer"):
        CoordinatorRepository(database_path, tmp_path / "other-run")

    legacy_path = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(legacy_path)
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute(
        "CREATE TABLE schema_migrations (version INTEGER PRIMARY KEY CHECK (version > 0), applied_at REAL NOT NULL)"
    )
    for version in range(1, 4):
        for statement in MIGRATIONS[version]:
            connection.execute(statement)
        connection.execute("INSERT INTO schema_migrations VALUES (?, 0)", (version,))
    connection.commit()
    connection.close()
    with CoordinatorRepository(legacy_path, tmp_path / "legacy-run") as upgraded:
        assert upgraded.connection.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 4
        assert "last_heartbeat_sent_at" in {
            row[1] for row in upgraded.connection.execute("PRAGMA table_info(worker_sessions)").fetchall()
        }
        assert {
            "request_id",
            "worker_session_id",
            "request_digest",
            "state",
            "lease_id",
            "created_at",
            "completed_at",
        } == {row[1] for row in upgraded.connection.execute("PRAGMA table_info(lease_requests)").fetchall()}
        scheduler_state = upgraded.connection.execute("SELECT * FROM scheduler_state").fetchone()
        assert scheduler_state["max_policy_lag"] is None
        assert scheduler_state["loaded_policy_preference_seconds"] is None


def test_run_policy_activation_and_worker_registration(tmp_path: Path):
    clock = FakeClock()
    with initialized_repository(tmp_path, clock) as state:
        state.initialize_run(base_policy())
        assert state.active_policy() == base_policy()

        policies_dir = state.run_root / "policies"
        trained = publish_lora_policy(
            policies_dir,
            run_id="run-1",
            policy_version=1,
            base_model=base_model(),
            state_dict={
                "model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4),
                "model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2),
            },
            rank=2,
            alpha=4,
            dropout=0,
            created_at=4.0,
        )
        state.record_policy(trained, policies_dir / trained.policy_id)
        state.record_policy(trained, policies_dir / trained.policy_id)
        assert state.activate_policy(trained.policy_id) == trained
        with pytest.raises(InvalidStateError, match="monotonic"):
            state.activate_policy(base_policy().policy_id)

        assert state.register_worker(registration()).created
        assert not state.register_worker(registration()).created
        conflicting = registration(caps=capabilities(capacity=2))
        with pytest.raises(ConflictError, match="capabilities"):
            state.register_worker(conflicting)
        other_model = base_model().model_copy(update={"model_config_digest": DIGEST_B})
        with pytest.raises(IncompatibleWorkerError, match="base model"):
            state.register_worker(registration(session_id="session-2", caps=capabilities(model=other_model)))

        policy_path = policies_dir / trained.policy_id
        (policy_path / "adapter_model.safetensors").write_bytes(b"corrupt")

    with pytest.raises(ArtifactCorruptionError, match="published policy"):
        repository(tmp_path, clock)


def test_group_validation_and_lease_capacity_renewal(tmp_path: Path):
    clock = FakeClock()
    with initialized_repository(tmp_path, clock) as state:
        state.register_worker(registration())
        group = assignments(base_policy(), size=2)
        with pytest.raises(ValueError, match="each group index"):
            state.create_group(group[:1], max_attempts=2)
        mixed = [group[0], group[1].model_copy(update={"task_data": {"prompt": "different"}})]
        with pytest.raises(ValueError, match="identical"):
            state.create_group(mixed, max_attempts=2)
        mixed_deadlines = [group[0], group[1].model_copy(update={"deadline_at": 200.0})]
        with pytest.raises(ValueError, match="identical"):
            state.create_group(mixed_deadlines, max_attempts=2)

        state.create_group(group, max_attempts=2)
        lease = state.create_lease(
            group[0].assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="lease-1",
            duration_seconds=5,
        )
        assert lease.expires_at == 15
        with pytest.raises(CapacityError):
            state.create_lease(
                group[1].assignment_id,
                worker_id="worker-1",
                worker_session_id="session-1",
                lease_id="lease-2",
                duration_seconds=5,
            )
        state.register_worker(registration(session_id="session-2"))
        state.create_lease(
            group[1].assignment_id,
            worker_id="worker-1",
            worker_session_id="session-2",
            lease_id="lease-2",
            duration_seconds=5,
        )
        clock.now = 12
        renewed = state.renew_lease("lease-1", worker_id="worker-1", worker_session_id="session-1", duration_seconds=10)
        assert renewed.expires_at == 22
        with pytest.raises(ConflictError):
            state.renew_lease("lease-1", worker_id="worker-1", worker_session_id="wrong", duration_seconds=2)

        clock.now = 16
        extra = assignments(base_policy(), group_id="group-2")[0]
        state.create_group([extra], max_attempts=1)
        state.create_lease(
            extra.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-2",
            lease_id="lease-3",
            duration_seconds=5,
        )


def test_fake_clock_expiry_retry_backoff_and_exhaustion(tmp_path: Path):
    clock = FakeClock()
    with initialized_repository(tmp_path, clock) as state:
        state.register_worker(registration())
        assignment = assignments(base_policy())[0]
        state.create_group([assignment], max_attempts=2)
        state.create_lease(
            assignment.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="lease-1",
            duration_seconds=2,
        )
        clock.now = 12
        assert state.expire_leases() == 1
        assert state.assignment_state(assignment.assignment_id) == "retry_wait"
        with pytest.raises(InvalidStateError, match="not due"):
            state.create_lease(
                assignment.assignment_id,
                worker_id="worker-1",
                worker_session_id="session-1",
                lease_id="too-early",
                duration_seconds=2,
            )
        clock.now = 14
        state.create_lease(
            assignment.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="lease-2",
            duration_seconds=2,
        )
        clock.now = 16
        state.expire_leases()
        assert state.assignment_state(assignment.assignment_id) == "failed"
        assert state.group_state(assignment.group_id) == "ready"

        deadline_assignment = assignments(base_policy(), group_id="deadline-group", deadline_at=16)[0]
        state.create_group([deadline_assignment], max_attempts=2)
        with pytest.raises(InvalidStateError, match="deadline"):
            state.create_lease(
                deadline_assignment.assignment_id,
                worker_id="worker-1",
                worker_session_id="session-1",
                lease_id="deadline-lease",
                duration_seconds=2,
            )
        assert state.assignment_state(deadline_assignment.assignment_id) == "failed"


def test_atomic_result_idempotency_conflict_and_processing_recovery(tmp_path: Path):
    clock = FakeClock()
    with initialized_repository(tmp_path, clock) as state:
        state.register_worker(registration())
        assignment = assignments(base_policy())[0]
        state.create_group([assignment], max_attempts=1)
        lease = state.create_lease(
            assignment.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="lease-1",
            duration_seconds=2,
        )
        envelope = result_envelope(lease)
        accepted = state.accept_result(envelope)
        assert not accepted.duplicate
        artifact = state.run_root / "spool" / "results" / f"{accepted.envelope_digest.removeprefix('sha256:')}.json"
        assert artifact.read_bytes() == canonical_json_bytes(envelope)
        assert state.group_state(assignment.group_id) == "ready"

        clock.now = 1000
        artifact.unlink()
        assert state.accept_result(envelope).duplicate
        assert artifact.read_bytes() == canonical_json_bytes(envelope)
        with pytest.raises(ConflictError, match="different accepted result"):
            state.accept_result(envelope.model_copy(update={"completed_at": 12.0}))

        claimed = state.claim_pending_results(1)
        assert claimed[0].path == artifact
        incoming = state.run_root / "spool" / "incoming" / "abandoned.tmp"
        incoming.write_bytes(b"partial")
        orphan = state.run_root / "spool" / "results" / "orphan.json"
        orphan.write_bytes(b"orphan")
        state.recover()
        assert not incoming.exists()
        assert not orphan.exists()
        assert state.claim_pending_results(1)[0].assignment_id == assignment.assignment_id
        state.mark_result_processed(assignment.assignment_id)


def test_failure_retry_idempotency_and_result_corruption_is_fatal(tmp_path: Path):
    clock = FakeClock()
    with initialized_repository(tmp_path, clock) as state:
        state.register_worker(registration())
        failure_assignment, result_assignment = assignments(base_policy(), size=2)
        state.create_group([failure_assignment, result_assignment], max_attempts=2)
        first_lease = state.create_lease(
            failure_assignment.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="failure-lease-1",
            duration_seconds=5,
        )
        failure = failure_envelope(first_lease)
        assert not state.accept_failure(failure).terminal
        assert state.accept_failure(failure).duplicate
        with pytest.raises(ConflictError, match="different failure"):
            state.accept_failure(failure_envelope(first_lease, message="different"))

        clock.now = 12
        second_lease = state.create_lease(
            failure_assignment.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="failure-lease-2",
            duration_seconds=5,
        )
        assert state.accept_failure(failure_envelope(second_lease, retryable=False)).terminal
        assert not state.accept_failure(failure).terminal

        result_lease = state.create_lease(
            result_assignment.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="result-lease",
            duration_seconds=5,
        )
        accepted = state.accept_result(result_envelope(result_lease, completed_at=12))
        artifact = state.run_root / "spool" / "results" / f"{accepted.envelope_digest.removeprefix('sha256:')}.json"
        artifact.write_bytes(b"corrupt")
        with pytest.raises(ArtifactCorruptionError, match="corrupt"):
            state.recover()


def test_spool_rejects_malformed_digest_and_symlinked_directory(tmp_path: Path):
    run_root = tmp_path / "run"
    run_root.mkdir()
    spool = AtomicSpool(run_root)
    with pytest.raises(ValueError, match="SHA-256"):
        spool.publish_result("../../escape", b"escaped")
    assert not (run_root / "escape.json").exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    spool.results_dir.rmdir()
    spool.results_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        spool.publish_result("sha256:" + "a" * 64, b"payload")
