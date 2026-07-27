import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest
import torch

from aether_rl.coordinator import CoordinatorRepository, create_coordinator_app
from aether_rl.protocol import (
    AssignmentLease,
    LeaseRequest,
    WorkerHeartbeat,
    canonical_json_bytes,
)
from aether_rl.trainer.policy import publish_lora_policy
from tests.unit.coordinator.test_database import (
    FakeClock,
    assignments,
    base_policy,
    capabilities,
    failure_envelope,
    registration,
    result_envelope,
)

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}", "Aether-Protocol-Version": "1"}
JSON_HEADERS = {**AUTH, "Content-Type": "application/json"}


def make_repository(tmp_path: Path, clock: FakeClock) -> CoordinatorRepository:
    run_root = tmp_path / "run"
    repository = CoordinatorRepository(run_root / "control" / "coordinator.sqlite3", run_root, clock=clock)
    repository.initialize_run(base_policy())
    return repository


@pytest.mark.asyncio
async def test_health_readiness_auth_protocol_registration_lease_and_status(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        app = create_coordinator_app(repository, token=TOKEN, trainer_ready=lambda: True)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/health")).json() == {"status": "ok"}
            assert (await client.get("/ready")).json() == {"status": "ready"}

            missing = await client.get("/api/v1/status")
            wrong = await client.get("/api/v1/status", headers={"Authorization": "Bearer wrong"})
            assert missing.status_code == wrong.status_code == 401
            assert missing.json() == wrong.json()
            assert missing.headers["www-authenticate"] == "Bearer"
            assert (
                await client.get("/api/v1/status", headers={**AUTH, "Authorization": "bearer test-token"})
            ).status_code == 200
            assert (await client.get("/api/v1/status", headers={"Authorization": f"Bearer {TOKEN}"})).status_code == 400

            body = canonical_json_bytes(registration())
            created = await client.post("/api/v1/workers/register", headers=JSON_HEADERS, content=body)
            retried = await client.post("/api/v1/workers/register", headers=JSON_HEADERS, content=body)
            assert created.status_code == retried.status_code == 200
            assert created.json()["created"] is True
            assert retried.json()["created"] is False
            conflicting = registration(caps=capabilities(capacity=2))
            conflict = await client.post(
                "/api/v1/workers/register", headers=JSON_HEADERS, content=canonical_json_bytes(conflicting)
            )
            assert conflict.status_code == 409

            lease_request = LeaseRequest(
                request_id="request-no-lease",
                worker_id="worker-1",
                worker_session_id="session-1",
                sent_at=clock.now,
                environments=registration().capabilities.environments,
                available_slots=1,
            )
            no_lease = await client.post(
                "/api/v1/assignments/lease", headers=JSON_HEADERS, content=canonical_json_bytes(lease_request)
            )
            assert no_lease.status_code == 204
            assert no_lease.headers["cache-control"] == "no-store"
            assert no_lease.headers["retry-after"] == "1"
            no_lease_retry = await client.post(
                "/api/v1/assignments/lease", headers=JSON_HEADERS, content=canonical_json_bytes(lease_request)
            )
            assert no_lease_retry.status_code == 204
            conflicting_request = lease_request.model_copy(update={"wait_seconds": 1})
            conflict = await client.post(
                "/api/v1/assignments/lease",
                headers=JSON_HEADERS,
                content=canonical_json_bytes(conflicting_request),
            )
            assert conflict.status_code == 409

            status = await client.get("/api/v1/status", headers=AUTH)
            assert status.status_code == 200
            assert status.json()["run_id"] == "run-1"
            assert status.json()["worker_sessions"] == 1
            assert status.json()["protocol_version"] == 1
            assert status.json()["trainer_ready"] is True
        app.state.coordinator_service.close()


@pytest.mark.asyncio
async def test_body_validation_limits_and_immediate_injected_lease(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_worker(registration(caps=capabilities(capacity=2)))
        assignment = assignments(base_policy())[0]
        repository.create_group([assignment], max_attempts=1)
        offered = repository.create_lease(
            assignment.assignment_id,
            lease_id="offered",
            worker_id="worker-1",
            worker_session_id="session-1",
            duration_seconds=10,
        )

        class Provider:
            async def try_lease(self, request: LeaseRequest) -> AssignmentLease | None:
                return offered

        app = create_coordinator_app(repository, token=TOKEN, lease_provider=Provider(), control_body_limit_bytes=1000)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            unsupported = await client.post("/api/v1/workers/register", headers=AUTH, content=b"{}")
            encoded = await client.post(
                "/api/v1/workers/register", headers={**JSON_HEADERS, "Content-Encoding": "gzip"}, content=b"{}"
            )
            malformed = await client.post("/api/v1/workers/register", headers=JSON_HEADERS, content=b"{")
            unversioned = await client.post(
                "/api/v1/workers/register",
                headers=JSON_HEADERS,
                content=json.dumps(
                    {
                        key: value
                        for key, value in registration().model_dump(mode="json").items()
                        if key != "protocol_version"
                    }
                ),
            )
            oversized = await client.post("/api/v1/workers/register", headers=JSON_HEADERS, content=b" " * 1001)
            assert unsupported.status_code == encoded.status_code == 415
            assert malformed.status_code == 422
            assert unversioned.status_code == 400
            assert oversized.status_code == 413
        app.state.coordinator_service.close()

        app = create_coordinator_app(repository, token=TOKEN, lease_provider=Provider())
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            request = LeaseRequest(
                request_id="request-offered",
                worker_id="worker-1",
                worker_session_id="session-1",
                sent_at=clock.now + 1,
                environments=registration().capabilities.environments,
                available_slots=1,
                wait_seconds=30,
            )
            response = await client.post(
                "/api/v1/assignments/lease", headers=JSON_HEADERS, content=canonical_json_bytes(request)
            )
            assert response.status_code == 200
            assert response.json()["lease_id"] == "offered"
            duplicate = await client.post(
                "/api/v1/assignments/lease",
                headers=JSON_HEADERS,
                content=canonical_json_bytes(
                    request.model_copy(update={"request_id": "request-offered-again", "sent_at": clock.now + 2})
                ),
            )
            assert duplicate.status_code == 409
        app.state.coordinator_service.close()

        class SlowProvider:
            async def try_lease(self, request: LeaseRequest) -> AssignmentLease | None:
                await asyncio.sleep(1)
                return None

        app = create_coordinator_app(
            repository, token=TOKEN, lease_provider=SlowProvider(), max_lease_wait_seconds=0.01
        )
        started = time.monotonic()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            request = LeaseRequest(
                request_id="request-slow",
                worker_id="worker-1",
                worker_session_id="session-1",
                sent_at=clock.now + 3,
                environments=registration().capabilities.environments,
                available_slots=1,
                wait_seconds=30,
            )
            response = await client.post(
                "/api/v1/assignments/lease", headers=JSON_HEADERS, content=canonical_json_bytes(request)
            )
            assert response.status_code == 204
            assert time.monotonic() - started < 0.2
        app.state.coordinator_service.close()


@pytest.mark.asyncio
async def test_heartbeat_atomic_renewal_stop_and_renew_path(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_worker(registration())
        repository.register_worker(registration(worker_id="worker-2", session_id="session-2"))
        first = assignments(base_policy(), group_id="first")[0]
        second = assignments(base_policy(), group_id="second")[0]
        repository.create_group([first], max_attempts=2)
        repository.create_group([second], max_attempts=2)
        own = repository.create_lease(
            first.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="lease-a",
            duration_seconds=5,
        )
        repository.create_lease(
            second.assignment_id,
            worker_id="worker-2",
            worker_session_id="session-2",
            lease_id="lease-b",
            duration_seconds=5,
        )
        app = create_coordinator_app(repository, token=TOKEN, lease_duration_seconds=20)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            foreign = WorkerHeartbeat(
                worker_id="worker-1",
                worker_session_id="session-1",
                sent_at=clock.now,
                active_lease_ids=("lease-a", "lease-b"),
            )
            response = await client.post(
                "/api/v1/workers/heartbeat", headers=JSON_HEADERS, content=canonical_json_bytes(foreign)
            )
            assert response.status_code == 409
            assert (
                repository.connection.execute(
                    "SELECT expires_at FROM lease_attempts WHERE lease_id = 'lease-a'"
                ).fetchone()[0]
                == own.expires_at
            )

            heartbeat = foreign.model_copy(update={"active_lease_ids": ("lease-a", "missing")})
            response = await client.post(
                "/api/v1/workers/heartbeat", headers=JSON_HEADERS, content=canonical_json_bytes(heartbeat)
            )
            assert response.status_code == 200
            assert response.json()["renewals"][0]["expires_at"] == 30
            assert response.json()["stop_lease_ids"] == ["missing"]
            replay = await client.post(
                "/api/v1/workers/heartbeat", headers=JSON_HEADERS, content=canonical_json_bytes(heartbeat)
            )
            assert replay.status_code == 409

            renewal = {
                "protocol_version": 1,
                "assignment_id": "wrong",
                "lease_id": "lease-a",
                "worker_id": "worker-1",
                "worker_session_id": "session-1",
                "sent_at": clock.now,
            }
            mismatch = await client.post(
                f"/api/v1/assignments/{first.assignment_id}/renew",
                headers=JSON_HEADERS,
                content=json.dumps(renewal),
            )
            assert mismatch.status_code == 409

            clock.now = first.deadline_at
            renewal["assignment_id"] = first.assignment_id
            deadline = await client.post(
                f"/api/v1/assignments/{first.assignment_id}/renew",
                headers=JSON_HEADERS,
                content=json.dumps(renewal),
            )
            assert deadline.status_code == 409
        app.state.coordinator_service.close()


@pytest.mark.asyncio
async def test_chunked_result_failure_idempotency_and_temp_cleanup(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_worker(registration(caps=capabilities(capacity=2)))
        result_assignment, failure_assignment = assignments(base_policy(), size=2)
        repository.create_group([result_assignment, failure_assignment], max_attempts=2)
        result_lease = repository.create_lease(
            result_assignment.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="result-lease",
            duration_seconds=20,
        )
        failure_lease = repository.create_lease(
            failure_assignment.assignment_id,
            worker_id="worker-1",
            worker_session_id="session-1",
            lease_id="failure-lease",
            duration_seconds=20,
        )
        envelope = canonical_json_bytes(result_envelope(result_lease))

        async def chunks():
            yield envelope[:10]
            yield envelope[10:]

        limited_app = create_coordinator_app(repository, token=TOKEN, result_body_limit_bytes=len(envelope) - 1)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=limited_app), base_url="http://test") as client:
            too_large = await client.put(
                f"/api/v1/assignments/{result_assignment.assignment_id}/result",
                headers=JSON_HEADERS,
                content=envelope,
            )
            assert too_large.status_code == 413
            assert not tuple(repository.spool.incoming_dir.iterdir())
        limited_app.state.coordinator_service.close()

        app = create_coordinator_app(repository, token=TOKEN)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            accepted = await client.put(
                f"/api/v1/assignments/{result_assignment.assignment_id}/result",
                headers=JSON_HEADERS,
                content=chunks(),
            )
            duplicate = await client.put(
                f"/api/v1/assignments/{result_assignment.assignment_id}/result",
                headers=JSON_HEADERS,
                content=envelope,
            )
            mismatch = await client.put(
                f"/api/v1/assignments/{failure_assignment.assignment_id}/result",
                headers=JSON_HEADERS,
                content=envelope,
            )
            assert accepted.status_code == 200
            assert duplicate.status_code == 200 and duplicate.json()["duplicate"] is True
            assert mismatch.status_code == 409
            assert not tuple(repository.spool.incoming_dir.iterdir())

            failure = canonical_json_bytes(failure_envelope(failure_lease))
            first = await client.post(
                f"/api/v1/assignments/{failure_assignment.assignment_id}/failure",
                headers=JSON_HEADERS,
                content=failure,
            )
            retry = await client.post(
                f"/api/v1/assignments/{failure_assignment.assignment_id}/failure",
                headers=JSON_HEADERS,
                content=failure,
            )
            assert first.status_code == 200
            assert retry.status_code == 200 and retry.json()["duplicate"] is True

            small = assignments(base_policy(), group_id="small")[0].model_copy(update={"result_size_limit_bytes": 16})
            repository.create_group([small], max_attempts=1)
            small_lease = repository.create_lease(
                small.assignment_id,
                worker_id="worker-1",
                worker_session_id="session-1",
                lease_id="small-lease",
                duration_seconds=20,
            )
            assignment_limited = await client.put(
                f"/api/v1/assignments/{small.assignment_id}/result",
                headers=JSON_HEADERS,
                content=canonical_json_bytes(result_envelope(small_lease)),
            )
            assert assignment_limited.status_code == 413
            assert not tuple(repository.spool.incoming_dir.iterdir())
        app.state.coordinator_service.close()


@pytest.mark.asyncio
async def test_policy_etags_allowlist_corruption_and_trainer_readiness(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        policies_dir = repository.run_root / "policies"
        trained = publish_lora_policy(
            policies_dir,
            run_id="run-1",
            policy_version=1,
            base_model=base_policy().base_model,
            state_dict={
                "model.layers.0.self_attn.q_proj.lora_A.weight": torch.ones(2, 4),
                "model.layers.0.self_attn.q_proj.lora_B.weight": torch.ones(4, 2),
            },
            rank=2,
            alpha=4,
            dropout=0,
            created_at=4,
        )
        repository.record_policy(trained, policies_dir / trained.policy_id)
        repository.activate_policy(trained.policy_id)
        app = create_coordinator_app(
            repository,
            token=TOKEN,
            trainer_ready=lambda: True,
            policy_verification_interval_seconds=0.01,
        )
        lifespan = app.router.lifespan_context(app)
        await lifespan.__aenter__()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.get("/ready")).status_code == 200
            current = await client.get("/api/v1/policies/current", headers=AUTH)
            assert current.status_code == 200
            assert current.headers["cache-control"] == "no-store"

            manifest = await client.get(f"/api/v1/policies/{trained.policy_id}/manifest", headers=AUTH)
            assert manifest.status_code == 200
            assert manifest.headers["etag"].startswith('"sha256:')
            not_modified = await client.get(
                f"/api/v1/policies/{trained.policy_id}/manifest",
                headers={**AUTH, "If-None-Match": manifest.headers["etag"]},
            )
            assert not_modified.status_code == 304
            wildcard = await client.get(
                f"/api/v1/policies/{trained.policy_id}/manifest",
                headers={**AUTH, "If-None-Match": "*"},
            )
            assert wildcard.status_code == 304
            weak_list = await client.get(
                f"/api/v1/policies/{trained.policy_id}/manifest",
                headers={**AUTH, "If-None-Match": f'"other", W/{manifest.headers["etag"]}'},
            )
            assert weak_list.status_code == 304

            file_url = f"/api/v1/policies/{trained.policy_id}/files/adapter_config.json"
            policy_file = await client.get(file_url, headers=AUTH)
            assert policy_file.status_code == 200
            assert policy_file.headers["content-type"].startswith("application/json")
            assert (
                await client.get(file_url, headers={**AUTH, "If-None-Match": policy_file.headers["etag"]})
            ).status_code == 304
            partial = await client.get(file_url, headers={**AUTH, "Range": "bytes=0-9"})
            assert partial.status_code == 206
            assert len(partial.content) == 10
            assert partial.headers["content-range"].startswith("bytes 0-9/")
            assert (
                await client.get(f"/api/v1/policies/{trained.policy_id}/files/manifest.json", headers=AUTH)
            ).status_code == 404

            config_path = policies_dir / trained.policy_id / "adapter_config.json"
            contents = config_path.read_bytes()
            config_path.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])
            corrupt = await client.get(file_url, headers=AUTH)
            assert corrupt.status_code == 500
            assert str(repository.run_root) not in corrupt.text
            await asyncio.sleep(0.03)
            assert (await client.get("/ready")).status_code == 503
        await lifespan.__aexit__(None, None, None)
