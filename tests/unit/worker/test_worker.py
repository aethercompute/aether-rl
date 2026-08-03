import asyncio
import json
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from aether_rl.configs.worker import WorkerConfig
from aether_rl.protocol import (
    InferenceExchangeResponse,
    InferenceLease,
    InferenceRequest,
)
from aether_rl.worker.daemon import ActiveInference, WorkerDaemon, build_registration
from tests.unit.coordinator.test_database import base_model, base_policy, registration


def worker_config(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig.model_validate(
        {
            "coordinator_url": "https://coordinator.example.com",
            "state_dir": tmp_path / "worker",
            "base_model": base_model().model_dump(mode="python"),
            "inference_slots": 1,
            "tensor_parallel_size": 1,
            "heartbeat_interval_seconds": 0.01,
            "lease_wait_seconds": 0,
            "request_timeout_seconds": 1,
            "retry_min_seconds": 0.001,
            "retry_max_seconds": 0.01,
            "shutdown_grace_seconds": 1,
        }
    )


def inference_lease() -> InferenceLease:
    return InferenceLease(
        assignment_id="assignment-1",
        lease_id="lease-1",
        attempt=1,
        worker_id="worker-1",
        worker_session_id="session-1",
        issued_at=3,
        expires_at=100,
        policy=base_policy(),
    )


def test_worker_config_and_registration_are_inference_only(tmp_path: Path):
    config = worker_config(tmp_path)
    discovered = build_registration(config, "worker-1", "session-1", gpu_count=1)
    assert discovered.capabilities.inference_slots == 1
    assert "environments" not in discovered.capabilities.model_dump()
    with pytest.raises(ValidationError):
        WorkerConfig.model_validate(
            config.model_dump(mode="python") | {"environments": [{"id": "secret"}], "execution_slots": 1}
        )


class ExchangeClient:
    def __init__(self):
        self.requests = []

    async def inference_exchange(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            return InferenceExchangeResponse(
                action="request",
                request=InferenceRequest(
                    request_id="inference-1",
                    method="POST",
                    path="/inference/v1/generate",
                    headers={"content-type": "application/json"},
                    body=b'{"prompt_token_ids":[1,2]}',
                ),
            )
        return InferenceExchangeResponse(action="stop")


@pytest.mark.asyncio
async def test_worker_proxies_only_inference_exchange_to_loopback(tmp_path: Path):
    local_requests = []

    async def local(request: httpx.Request) -> httpx.Response:
        local_requests.append(request)
        return httpx.Response(200, json={"token_ids": [3], "logprobs": [0.0]})

    exchange = ExchangeClient()
    local_client = httpx.AsyncClient(
        transport=httpx.MockTransport(local),
        base_url="http://127.0.0.1:8000",
    )
    daemon = WorkerDaemon(
        worker_config(tmp_path),
        registration(),
        exchange,  # type: ignore[arg-type]
        inference_client=local_client,
    )
    active = ActiveInference(inference_lease(), asyncio.Event(), 100)
    await daemon._serve_lease(active)
    await local_client.aclose()

    assert local_requests[0].url.path == "/inference/v1/generate"
    assert json.loads(local_requests[0].content) == {"prompt_token_ids": [1, 2]}
    reply = exchange.requests[1].reply
    assert reply is not None
    assert reply.request_id == "inference-1"
    assert reply.status_code == 200
