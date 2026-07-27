import asyncio
import threading
import time
from pathlib import Path

import httpx
import pytest
import torch
from pydantic import ValidationError
from verifiers.v1.types import SamplingConfig

from aether_rl.coordinator import (
    ConflictError,
    CoordinatorRepository,
    EnvironmentCatalog,
    EnvironmentSourceSpec,
    InvalidStateError,
    create_coordinator_app,
    verifier_v1_task_payloads,
)
from aether_rl.protocol import EnvironmentIdentity, LeaseRequest, WorkerHeartbeat, canonical_json_bytes
from aether_rl.trainer.policy import publish_lora_policy
from tests.unit.coordinator.test_database import FakeClock, base_policy, capabilities, registration


def make_repository(tmp_path: Path, clock: FakeClock) -> CoordinatorRepository:
    run_root = tmp_path / "run"
    repository = CoordinatorRepository(
        run_root / "control" / "coordinator.sqlite3",
        run_root,
        clock=clock,
        retry_base_seconds=2,
        retry_max_seconds=8,
    )
    repository.initialize_run(base_policy())
    repository.configure_scheduler(max_policy_lag=0, loaded_policy_preference_seconds=5)
    return repository


def source(
    source_id: str,
    *,
    kind: str = "train",
    environment: EnvironmentIdentity = EnvironmentIdentity(id="env", revision="1"),
    weight: float = 1,
    size: int = 1,
    attempts: int = 2,
) -> EnvironmentSourceSpec:
    return EnvironmentSourceSpec(
        source_id=source_id,
        kind=kind,
        environment=environment,
        tasks=({"task": 0}, {"task": 1}),
        sampling=SamplingConfig(max_tokens=8),
        group_size=size,
        max_attempts=attempts,
        result_size_limit_bytes=1024,
        assignment_timeout_seconds=100,
        weight=weight,
    )


def lease_request(
    clock: FakeClock,
    *,
    request_id: str,
    sent_at: float | None = None,
    environments: tuple[EnvironmentIdentity, ...] = (EnvironmentIdentity(id="env", revision="1"),),
    loaded_policy_ids: tuple[str, ...] = (),
    slots: int = 1,
    wait_seconds: float = 0,
) -> LeaseRequest:
    return LeaseRequest(
        request_id=request_id,
        worker_id="worker-1",
        worker_session_id="session-1",
        sent_at=clock.now if sent_at is None else sent_at,
        environments=environments,
        loaded_policy_ids=loaded_policy_ids,
        available_slots=slots,
        wait_seconds=wait_seconds,
    )


def publish_policy(repository: CoordinatorRepository, version: int, created_at: float):
    policies_dir = repository.run_root / "policies"
    manifest = publish_lora_policy(
        policies_dir,
        run_id="run-1",
        policy_version=version,
        base_model=base_policy().base_model,
        state_dict={
            "model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4) * version,
            "model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2) * version,
        },
        rank=2,
        alpha=4,
        dropout=0,
        created_at=created_at,
    )
    repository.record_policy(manifest, policies_dir / manifest.policy_id)
    return manifest


def test_catalog_is_pure_validated_and_registration_conflicts(tmp_path: Path):
    spec = source("source-a")
    assert EnvironmentCatalog(sources=(spec,)).sources == (spec,)
    with pytest.raises(ValidationError):
        EnvironmentSourceSpec.model_validate(spec.model_dump(mode="python") | {"tasks": ()})
    with pytest.raises(ValidationError):
        EnvironmentSourceSpec.model_validate(spec.model_dump(mode="python") | {"weight": float("inf")})

    class Data:
        def model_dump(self, *, mode: str):
            assert mode == "json"
            return {"already": "loaded"}

    class Task:
        data = Data()

    assert verifier_v1_task_payloads([Task()]) == ({"already": "loaded"},)

    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_scheduler_source(spec)
        repository.register_scheduler_source(spec)
        with pytest.raises(ConflictError, match="different immutable"):
            repository.register_scheduler_source(spec.model_copy(update={"group_size": 2}))


def test_weighted_generation_is_atomic_deterministic_and_survives_restart(tmp_path: Path):
    clock = FakeClock()
    ids = iter(f"id-{index}" for index in range(100))
    with make_repository(tmp_path, clock) as repository:
        repository.register_scheduler_source(source("a", weight=2, size=2))
        repository.register_scheduler_source(source("b", weight=1))
        records = [repository.create_next_group("train", lambda: next(ids)) for _ in range(5)]
        assert [record.source_id for record in records] == ["a", "b", "a", "b", "a"]
        assert [record.sequence for record in records] == [1, 2, 3, 4, 5]
        assert [item.group_index for item in records[0].assignments] == [0, 1]
        assert all(item.policy == base_policy() for record in records for item in record.assignments)
        assert records[0].assignments[0].task_data == {"task": 0}
        assert records[2].assignments[0].task_data == {"task": 1}
        assert repository.connection.execute("PRAGMA foreign_key_check").fetchall() == []

    with make_repository(tmp_path, clock) as reopened:
        record = reopened.create_next_group("train", lambda: next(ids))
        assert record.source_id == "b"
        assert record.source_cursor == 2
        assert record.sequence == 6
        assert record.assignments[0].task_data == {"task": 0}
        reopened.register_scheduler_source(source("c"))
        dynamic = [reopened.create_next_group("train", lambda: next(ids)) for _ in range(3)]
        assert [item.source_id for item in dynamic] == ["a", "b", "c"]
        reopened.register_scheduler_source(source("eval", kind="eval"))
        mixed = [reopened.create_next_group(None, lambda: next(ids)) for _ in range(4)]
        assert [item.source_id for item in mixed] == ["a", "b", "c", "eval"]


def test_compatible_leasing_preference_order_capacity_expiry_and_secure_ids(tmp_path: Path):
    clock = FakeClock()
    other = EnvironmentIdentity(id="other", revision="2")
    caps = capabilities(capacity=2).model_copy(
        update={"environments": (EnvironmentIdentity(id="env", revision="1"), other)}
    )
    with make_repository(tmp_path, clock) as repository:
        repository.register_worker(registration(caps=caps))
        repository.register_scheduler_source(source("a-other", environment=other))
        repository.register_scheduler_source(source("b-env"))
        other_group = repository.create_next_group("train")
        env_group = repository.create_next_group("train")

        repository.configure_scheduler(max_policy_lag=0, loaded_policy_preference_seconds=0)
        request = lease_request(
            clock, request_id="request-env-1", environments=(EnvironmentIdentity(id="env", revision="1"),)
        )
        repository.validate_lease_request(request)
        first = repository.lease_next_compatible(request, lease_duration_seconds=2)
        assert first.assignment.group_id == env_group.group_id
        assert first.lease_id.startswith("lease-") and len(first.lease_id) == 70
        other_request = lease_request(clock, request_id="request-other", sent_at=11, environments=(other,))
        repository.validate_lease_request(other_request)
        second = repository.lease_next_compatible(other_request, lease_duration_seconds=2)
        assert second.assignment.group_id == other_group.group_id
        assert second.lease_id != first.lease_id
        retry_same_request = repository.lease_next_compatible(request, lease_duration_seconds=2)
        assert retry_same_request == first

        clock.now = 12
        repository.expire_leases()
        assert repository.assignment_state(first.assignment.assignment_id) == "retry_wait"
        fresh_request = lease_request(clock, request_id="request-env-fresh", sent_at=12)
        repository.validate_lease_request(fresh_request)
        fresh = repository.lease_or_create_next_compatible(fresh_request, lease_duration_seconds=2)
        assert fresh.assignment.assignment_id != first.assignment.assignment_id
        clock.now = 14
        retry_request = lease_request(clock, request_id="request-env-2", sent_at=14)
        repository.validate_lease_request(retry_request)
        retry = repository.lease_next_compatible(retry_request, lease_duration_seconds=2)
        assert retry.assignment.assignment_id == first.assignment.assignment_id
        assert retry.attempt == 2


def test_loaded_preference_staleness_cancellation_eval_exemption_and_restart(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_worker(registration(caps=capabilities(capacity=1)))
        repository.register_scheduler_source(source("train"))
        old_group = repository.create_next_group("train")
        policy1 = publish_policy(repository, 1, 4)
        repository.configure_scheduler(max_policy_lag=1, loaded_policy_preference_seconds=5)
        repository.activate_policy(policy1.policy_id)
        loaded_group = repository.create_next_group("train")

        preferred_request = lease_request(clock, request_id="request-preferred", loaded_policy_ids=(policy1.policy_id,))
        repository.validate_lease_request(preferred_request)
        preferred = repository.lease_next_compatible(preferred_request, lease_duration_seconds=2)
        assert preferred.assignment.group_id == loaded_group.group_id
        clock.now = 16
        repository.expire_leases()
        bounded_request = lease_request(
            clock, request_id="request-bounded", sent_at=16, loaded_policy_ids=(policy1.policy_id,)
        )
        repository.validate_lease_request(bounded_request)
        bounded = repository.lease_next_compatible(bounded_request, lease_duration_seconds=2)
        assert bounded.assignment.group_id == old_group.group_id

        policy2 = publish_policy(repository, 2, 5)
        repository.activate_policy(policy2.policy_id)
        assert repository.cancellation_state(old_group.assignments[0].assignment_id) == ("policy_stale", False)
        policy3 = publish_policy(repository, 3, 6)
        repository.activate_policy(policy3.policy_id)
        assert repository.assignment_state(loaded_group.assignments[0].assignment_id) == "failed"
        assert repository.cancellation_state(loaded_group.assignments[0].assignment_id) == ("policy_stale", True)
        renewals, stop_ids = repository.record_heartbeat(
            WorkerHeartbeat(
                worker_id="worker-1",
                worker_session_id="session-1",
                sent_at=clock.now,
                active_lease_ids=(bounded.lease_id,),
            ),
            duration_seconds=10,
        )
        assert renewals == ()
        assert stop_ids == (bounded.lease_id,)
        repository.register_scheduler_source(source("eval", kind="eval"))
        eval_group = repository.create_next_group("eval")
        eval_request = lease_request(clock, request_id="request-eval", sent_at=17)
        repository.validate_lease_request(eval_request)
        assert (
            repository.lease_next_compatible(eval_request, lease_duration_seconds=2).assignment.group_id
            == eval_group.group_id
        )

    with make_repository(tmp_path, clock) as reopened:
        cancellation = reopened.cancellation_state(old_group.assignments[0].assignment_id)
        assert cancellation == ("policy_stale", False)
        with pytest.raises(InvalidStateError, match="cancelled"):
            reopened.renew_lease(
                bounded.lease_id,
                worker_id="worker-1",
                worker_session_id="session-1",
                duration_seconds=2,
            )
        clock.now = bounded.expires_at
        reopened.expire_leases()
        assert reopened.assignment_state(bounded.assignment.assignment_id) == "failed"
        assert reopened.cancellation_state(bounded.assignment.assignment_id) == ("policy_stale", True)
        assert reopened.group_state(old_group.group_id) == "ready"


@pytest.mark.asyncio
async def test_default_api_scheduler_returns_durable_lease(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_scheduler_source(source("api"))
        app = create_coordinator_app(repository, token="token")
        headers = {
            "Authorization": "Bearer token",
            "Aether-Protocol-Version": "1",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/api/v1/workers/register", headers=headers, content=canonical_json_bytes(registration()))
            response = await client.post(
                "/api/v1/assignments/lease",
                headers=headers,
                content=canonical_json_bytes(lease_request(clock, request_id="request-api")),
            )
            assert response.status_code == 200
            lease_id = response.json()["lease_id"]
            retry = await client.post(
                "/api/v1/assignments/lease",
                headers=headers,
                content=canonical_json_bytes(lease_request(clock, request_id="request-api")),
            )
            assert retry.status_code == 200
            assert retry.json() == response.json()
            conflict = await client.post(
                "/api/v1/assignments/lease",
                headers=headers,
                content=canonical_json_bytes(
                    lease_request(clock, request_id="request-api").model_copy(update={"wait_seconds": 1})
                ),
            )
            assert conflict.status_code == 409
            policy1 = publish_policy(repository, 1, 4)
            repository.activate_policy(policy1.policy_id)
            stopped = await client.post(
                f"/api/v1/assignments/{response.json()['assignment']['assignment_id']}/renew",
                headers=headers,
                content=canonical_json_bytes(
                    {
                        "protocol_version": 1,
                        "assignment_id": response.json()["assignment"]["assignment_id"],
                        "lease_id": lease_id,
                        "worker_id": "worker-1",
                        "worker_session_id": "session-1",
                        "sent_at": clock.now,
                    }
                ),
            )
            assert stopped.status_code == 200
            assert stopped.json()["action"] == "stop"
            assert stopped.json()["reason"] == "policy_stale"
            assert stopped.json()["renewal"] is None
        assert (
            repository.connection.execute(
                "SELECT state FROM lease_attempts WHERE lease_id = ?", (lease_id,)
            ).fetchone()["state"]
            == "active"
        )
        app.state.coordinator_service.close()


@pytest.mark.asyncio
async def test_cancelled_default_scheduler_request_recovers_committed_lease(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_scheduler_source(source("api-delayed"))
        app = create_coordinator_app(repository, token="token", durable_provider_timeout_seconds=0.01)
        started = threading.Event()
        first_completed = threading.Event()
        completed = threading.Event()
        original = repository.lease_or_create_next_compatible
        call_count = 0

        def delayed(request: LeaseRequest, *, lease_duration_seconds: float):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                started.set()
            time.sleep(0.05)
            lease = original(request, lease_duration_seconds=lease_duration_seconds)
            if call_count == 1:
                first_completed.set()
            else:
                completed.set()
            return lease

        repository.lease_or_create_next_compatible = delayed  # type: ignore[method-assign]
        headers = {
            "Authorization": "Bearer token",
            "Aether-Protocol-Version": "1",
            "Content-Type": "application/json",
        }
        timeout_body = canonical_json_bytes(lease_request(clock, request_id="request-timeout", wait_seconds=30))
        cancelled_body = canonical_json_bytes(
            lease_request(clock, request_id="request-cancelled", sent_at=11, wait_seconds=30)
        )
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            await client.post(
                "/api/v1/workers/register",
                headers=headers,
                content=canonical_json_bytes(registration(caps=capabilities(capacity=2))),
            )
            timed_out = await client.post("/api/v1/assignments/lease", headers=headers, content=timeout_body)
            assert timed_out.status_code == 503
            assert timed_out.json()["error"]["code"] == "lease_pending"
            await asyncio.to_thread(first_completed.wait)
            first_recovered = await client.post("/api/v1/assignments/lease", headers=headers, content=timeout_body)
            assert first_recovered.status_code == 200
            pending = asyncio.create_task(
                client.post("/api/v1/assignments/lease", headers=headers, content=cancelled_body)
            )
            await asyncio.to_thread(started.wait)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            await asyncio.to_thread(completed.wait)
            recovered = await client.post("/api/v1/assignments/lease", headers=headers, content=cancelled_body)
        assert recovered.status_code == 200
        assert repository.connection.execute("SELECT COUNT(*) FROM lease_attempts").fetchone()[0] == 2
        app.state.coordinator_service.close()
