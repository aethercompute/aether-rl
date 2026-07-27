import asyncio
import json
from pathlib import Path

import httpx
import pytest
import torch

from aether_rl.configs.worker import WorkerConfig
from aether_rl.protocol import canonical_json_bytes, sha256_digest
from aether_rl.trainer.policy import publish_lora_policy
from aether_rl.worker.client import CoordinatorClient
from aether_rl.worker.policy_cache import AdapterCache
from aether_rl.worker.policy_runtime import VLLMAdminClient, WorkerVLLMSupervisor
from aether_rl.worker.spool import WorkerState
from tests.unit.coordinator.test_database import base_model
from tests.unit.worker.test_worker import worker_config


def published_policy(tmp_path: Path):
    root = tmp_path / "published"
    manifest = publish_lora_policy(
        root,
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
        created_at=1,
    )
    return manifest, root / manifest.policy_id


@pytest.mark.asyncio
async def test_adapter_cache_downloads_verifies_and_reuses_concurrently(tmp_path: Path):
    manifest, source = published_policy(tmp_path)
    requests: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path.endswith("/manifest"):
            body = canonical_json_bytes(manifest)
            return httpx.Response(
                200,
                content=body,
                headers={"Content-Type": "application/json", "ETag": f'"{sha256_digest(body)}"'},
            )
        name = request.url.path.rsplit("/", 1)[-1]
        body = (source / name).read_bytes()
        artifact = next(item for item in manifest.adapter.files if item.name == name)
        content_type = "application/json" if name.endswith(".json") else "application/octet-stream"
        return httpx.Response(
            200,
            content=body,
            headers={
                "Content-Type": content_type,
                "Content-Length": str(len(body)),
                "ETag": f'"{artifact.digest}"',
            },
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://coordinator.test")
    coordinator = CoordinatorClient("https://coordinator.test", "token", timeout_seconds=5, client=async_client)
    with WorkerState(tmp_path / "worker") as state:
        cache = AdapterCache(state, coordinator, max_bytes=1024**3)
        first, second = await asyncio.gather(cache.ensure(manifest), cache.ensure(manifest))
        assert first == second
        assert first.path.name == manifest.policy_id
        assert sorted(path.name for path in first.path.iterdir()) == [
            "adapter_config.json",
            "adapter_model.safetensors",
            "manifest.json",
        ]
        assert requests.count(f"/api/v1/policies/{manifest.policy_id}/manifest") == 1
        assert len(requests) == 3
        assert await cache.ensure(manifest) == first
        assert len(requests) == 3
    await async_client.aclose()


@pytest.mark.asyncio
async def test_native_vllm_admin_never_uses_inplace_and_rejects_name_conflict(tmp_path: Path):
    loaded: dict[str, str | None] = {base_model().model_name: None}
    load_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": key, "root": value} for key, value in loaded.items()]})
        payload = json.loads(request.content)
        if request.url.path == "/v1/load_lora_adapter":
            load_payloads.append(payload)
            loaded[payload["lora_name"]] = payload["lora_path"]
            return httpx.Response(200)
        if request.url.path == "/v1/unload_lora_adapter":
            loaded.pop(payload["lora_name"], None)
            return httpx.Response(200)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://127.0.0.1:8000")
    admin = VLLMAdminClient(client)
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    await admin.load("policy-1", adapter_path)
    await admin.load("policy-1", adapter_path)
    assert load_payloads == [{"lora_name": "policy-1", "lora_path": str(adapter_path)}]
    assert "load_inplace" not in load_payloads[0]
    loaded["policy-1"] = str(tmp_path / "different")
    with pytest.raises(Exception, match="different adapter"):
        await admin.load("policy-1", adapter_path)
    await client.aclose()


@pytest.mark.asyncio
async def test_supervisor_config_is_loopback_and_pins_revisions(tmp_path: Path):
    config: WorkerConfig = worker_config(tmp_path)
    with WorkerState(config.state_dir) as state:
        supervisor = WorkerVLLMSupervisor(config, state)
        supervisor._write_config()
        contents = supervisor.config_path.read_text()
        assert 'host = "127.0.0.1"' in contents
        assert f'revision = "{config.base_model.model_revision}"' in contents
        assert f'name = "{config.base_model.tokenizer_name}"' in contents
        assert "enable_lora = true" in contents
        await supervisor.client.aclose()
