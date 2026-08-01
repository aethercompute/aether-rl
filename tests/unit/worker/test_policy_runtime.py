import asyncio
import json
import time
import tomllib
from pathlib import Path

import httpx
import pytest
import torch

from aether_rl.configs.inference import InferenceConfig
from aether_rl.configs.worker import WorkerConfig
from aether_rl.protocol import (
    PolicyFileLocation,
    PolicyLocations,
    canonical_json_bytes,
    policy_manifest_digest,
    sha256_digest,
)
from aether_rl.trainer.policy import publish_lora_policy
from aether_rl.worker.client import CoordinatorClient
from aether_rl.worker.policy_cache import AdapterCache
from aether_rl.worker.policy_runtime import VLLMAdminClient, WorkerVLLMSupervisor
from aether_rl.worker.policy_transport import PolicyFileTransport
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
async def test_adapter_cache_replaces_full_size_corrupt_partial(tmp_path: Path):
    manifest, source = published_policy(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
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
        return httpx.Response(
            200,
            content=body,
            headers={
                "Content-Type": "application/json" if name.endswith(".json") else "application/octet-stream",
                "Content-Length": str(len(body)),
                "ETag": f'"{artifact.digest}"',
            },
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://coordinator.test")
    coordinator = CoordinatorClient("https://coordinator.test", "token", timeout_seconds=5, client=async_client)
    with WorkerState(tmp_path / "worker-corrupt-partial") as state:
        cache = AdapterCache(state, coordinator, max_bytes=1024**3)
        weight = next(item for item in manifest.adapter.files if item.name == "adapter_model.safetensors")
        partial = cache.incoming / f"{weight.digest.removeprefix('sha256:')}.{weight.name}.part"
        partial.write_bytes(b"x" * weight.size_bytes)

        cached = await cache.ensure(manifest)

        assert cached is not None
        assert (cached.path / weight.name).read_bytes() == (source / weight.name).read_bytes()
    await async_client.aclose()


@pytest.mark.asyncio
async def test_adapter_cache_resumes_from_external_policy_location_without_forwarding_auth(tmp_path: Path):
    manifest, source = published_policy(tmp_path)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/manifest"):
            body = canonical_json_bytes(manifest)
            return httpx.Response(
                200,
                content=body,
                headers={"Content-Type": "application/json", "ETag": f'"{sha256_digest(body)}"'},
            )
        if request.url.path.endswith("/locations"):
            locations = PolicyLocations(
                policy_id=manifest.policy_id,
                policy_digest=policy_manifest_digest(manifest),
                expires_at=time.time() + 60,
                files=tuple(
                    PolicyFileLocation(name=item.name, url=f"https://cdn.test/files/{item.name}")
                    for item in manifest.adapter.files
                ),
            )
            return httpx.Response(
                200, content=canonical_json_bytes(locations), headers={"Content-Type": "application/json"}
            )
        name = request.url.path.rsplit("/", 1)[-1]
        body = (source / name).read_bytes()
        offset = 7 if request.headers.get("range") == "bytes=7-" else 0
        content_type = "application/json" if name.endswith(".json") else "application/octet-stream"
        if request.url.host == "cdn.test" and name == "adapter_config.json":
            content_type = "text/plain"
        headers = {
            "Content-Type": content_type,
            "Content-Length": str(len(body) - offset),
        }
        if request.url.host == "coordinator.test":
            artifact = next(item for item in manifest.adapter.files if item.name == name)
            headers["ETag"] = f'"{artifact.digest}"'
        status = 200
        if offset:
            status = 206
            headers["Content-Range"] = f"bytes {offset}-{len(body) - 1}/{len(body)}"
        return httpx.Response(status, content=body[offset:], headers=headers)

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://coordinator.test")
    config = worker_config(tmp_path).model_copy(update={"policy_download_allowed_origins": ["https://cdn.test"]})
    coordinator = CoordinatorClient("https://coordinator.test", "token", timeout_seconds=5, client=async_client)
    with WorkerState(tmp_path / "worker-external") as state:
        transport = PolicyFileTransport(config, coordinator)
        cache = AdapterCache(state, coordinator, max_bytes=1024**3, transport=transport)
        weight = next(item for item in manifest.adapter.files if item.name == "adapter_model.safetensors")
        partial = cache.incoming / f"{weight.digest.removeprefix('sha256:')}.{weight.name}.part"
        partial.write_bytes((source / weight.name).read_bytes()[:7])
        cached = await cache.ensure(manifest)
        assert cached is not None
        assert (cached.path / weight.name).read_bytes() == (source / weight.name).read_bytes()

    external = [request for request in requests if request.url.host == "cdn.test"]
    assert external
    assert all("authorization" not in request.headers for request in external)
    assert any(request.headers.get("range") == "bytes=7-" for request in external)
    assert any(
        request.url.host == "coordinator.test" and request.url.path.endswith("/files/adapter_config.json")
        for request in requests
    )
    assert sum(request.url.path.endswith("/adapter_config.json") for request in external) == 2
    await async_client.aclose()


@pytest.mark.asyncio
async def test_adapter_cache_uses_shardcast_before_coordinator_file_download(tmp_path: Path):
    manifest, source = published_policy(tmp_path)
    weight = next(item for item in manifest.adapter.files if item.name == "adapter_model.safetensors")
    weight_bytes = (source / weight.name).read_bytes()
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
        if request.url.path == "/aether-policies.json":
            return httpx.Response(
                200,
                json={
                    "version": 1,
                    "policies": {
                        manifest.policy_id: {
                            "policy_digest": policy_manifest_digest(manifest),
                            "artifact_digest": weight.digest,
                            "size_bytes": len(weight_bytes),
                            "version": "v7",
                            "shard_count": 1,
                            "shard_size_bytes": 1024**2,
                        }
                    },
                },
            )
        if request.url.path == "/v7/shard_00001.bin":
            return httpx.Response(200, content=weight_bytes)
        name = request.url.path.rsplit("/", 1)[-1]
        body = (source / name).read_bytes()
        artifact = next(item for item in manifest.adapter.files if item.name == name)
        return httpx.Response(
            200,
            content=body,
            headers={
                "Content-Type": "application/json" if name.endswith(".json") else "application/octet-stream",
                "Content-Length": str(len(body)),
                "ETag": f'"{artifact.digest}"',
            },
        )

    async_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://coordinator.test")
    config = worker_config(tmp_path).model_copy(update={"shardcast_servers": ["https://relay.test"]})
    coordinator = CoordinatorClient("https://coordinator.test", "token", timeout_seconds=5, client=async_client)
    with WorkerState(tmp_path / "worker-shardcast") as state:
        cache = AdapterCache(
            state,
            coordinator,
            max_bytes=1024**3,
            transport=PolicyFileTransport(config, coordinator),
        )
        cached = await cache.ensure(manifest)
        assert cached is not None
        assert (cached.path / weight.name).read_bytes() == weight_bytes
    assert f"/api/v1/policies/{manifest.policy_id}/files/{weight.name}" not in requests
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
    config: WorkerConfig = worker_config(tmp_path).model_copy(
        update={
            "max_model_len": 32768,
            "enable_prefix_caching": True,
            "enable_dbo": True,
            "enable_chunked_prefill": True,
            "gpu_memory_utilization": 0.82,
            "quantization": "fp8",
            "vllm_extra": {"max_num_batched_tokens": 65536},
        }
    )
    with WorkerState(config.state_dir) as state:
        supervisor = WorkerVLLMSupervisor(config, state)
        supervisor._write_config()
        contents = supervisor.config_path.read_text()
        assert 'host = "127.0.0.1"' in contents
        assert f'revision = "{config.base_model.model_revision}"' in contents
        assert f'name = "{config.base_model.tokenizer_name}"' in contents
        assert "enable_lora = true" in contents
        assert "max_model_len = 32768" in contents
        inference = InferenceConfig.model_validate(tomllib.loads(contents))
        assert inference.model.max_model_len == 32768
        assert inference.enable_prefix_caching is True
        assert inference.enable_dbo is True
        assert inference.gpu_memory_utilization == 0.82
        assert inference.quantization == "fp8"
        assert inference.vllm_extra == {
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 65536,
        }
        await supervisor.client.aclose()
