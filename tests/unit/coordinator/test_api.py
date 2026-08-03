from pathlib import Path

import httpx
import pytest

from aether_rl.coordinator import create_coordinator_app
from aether_rl.protocol import LeaseRequest, canonical_json_bytes
from tests.unit.coordinator.test_database import FakeClock, registration
from tests.unit.coordinator.test_scheduler import make_repository, source

TOKEN = "test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}", "Aether-Protocol-Version": "2"}
JSON_HEADERS = {**AUTH, "Content-Type": "application/json"}


@pytest.mark.asyncio
async def test_authenticated_lease_is_redacted_and_protocol_is_breaking(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        repository.register_scheduler_source(source("source"))
        app = create_coordinator_app(repository, token=TOKEN)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                registered = await client.post(
                    "/api/v2/workers/register",
                    headers=JSON_HEADERS,
                    content=canonical_json_bytes(registration()),
                )
                assert registered.status_code == 200
                request = LeaseRequest(
                    request_id="request-1",
                    worker_id="worker-1",
                    worker_session_id="session-1",
                    sent_at=clock.now,
                    available_slots=1,
                )
                response = await client.post(
                    "/api/v2/assignments/lease",
                    headers=JSON_HEADERS,
                    content=canonical_json_bytes(request),
                )
                assert response.status_code == 200
                lease = response.json()
                assert set(lease) == {
                    "protocol_version",
                    "assignment_id",
                    "lease_id",
                    "attempt",
                    "worker_id",
                    "worker_session_id",
                    "issued_at",
                    "expires_at",
                    "policy",
                }
                assert "task_data" not in response.text
                old_protocol = await client.post(
                    "/api/v2/assignments/lease",
                    headers=JSON_HEADERS | {"Aether-Protocol-Version": "1"},
                    content=canonical_json_bytes(request),
                )
                assert old_protocol.status_code == 400
        finally:
            app.state.coordinator_service.close()


@pytest.mark.asyncio
async def test_worker_result_and_group_upload_routes_are_removed(tmp_path: Path):
    clock = FakeClock()
    with make_repository(tmp_path, clock) as repository:
        app = create_coordinator_app(repository, token=TOKEN)
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                for method, path in (
                    ("post", "/api/v2/assignments/lease-group"),
                    ("put", "/api/v2/assignments/assignment-1/result"),
                    ("post", "/api/v2/assignments/assignment-1/failure"),
                ):
                    response = await getattr(client, method)(path, headers=JSON_HEADERS, content=b"{}")
                    assert response.status_code == 404
        finally:
            app.state.coordinator_service.close()
