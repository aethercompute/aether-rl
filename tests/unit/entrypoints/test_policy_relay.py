import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import torch

from aether_rl.configs.policy_relay import PolicyRelayConfig
from aether_rl.entrypoints.policy_relay import ShardcastPolicyRelay
from aether_rl.protocol import PolicyFileLocation, PolicyLocations, policy_manifest_digest
from aether_rl.trainer.policy import publish_lora_policy
from aether_rl.worker.client import CoordinatorClient
from aether_rl.worker.spool import WorkerState
from tests.unit.coordinator.test_database import base_model


@pytest.mark.asyncio
async def test_policy_relay_broadcasts_verified_external_adapter_once(tmp_path: Path, monkeypatch):
    published = tmp_path / "published"
    manifest = publish_lora_policy(
        published,
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
    artifact = next(file for file in manifest.adapter.files if file.name == "adapter_model.safetensors")
    artifact_bytes = (published / manifest.policy_id / artifact.name).read_bytes()
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/policies/current":
            return httpx.Response(200, json=manifest.model_dump(mode="json"))
        if request.url.path.endswith("/locations"):
            locations = PolicyLocations(
                policy_id=manifest.policy_id,
                policy_digest=policy_manifest_digest(manifest),
                expires_at=60,
                files=tuple(
                    PolicyFileLocation(name=file.name, url=f"https://cdn.test/{file.name}")
                    for file in manifest.adapter.files
                ),
            )
            return httpx.Response(200, json=locations.model_dump(mode="json"))
        if request.url == httpx.URL(f"https://cdn.test/{artifact.name}"):
            return httpx.Response(200, content=artifact_bytes)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="https://coordinator.test")
    coordinator = CoordinatorClient("https://coordinator.test", "token", timeout_seconds=5, client=client)
    config = PolicyRelayConfig(
        coordinator_url="https://coordinator.test",
        state_dir=tmp_path / "relay",
        policy_download_allowed_origins=["https://cdn.test"],
    )
    broadcasts: list[bytes] = []

    with WorkerState(config.state_dir) as state:
        relay = ShardcastPolicyRelay(config, coordinator, state)

        def broadcast(path: str, shard_size: int) -> str:
            broadcasts.append(Path(path).read_bytes())
            (relay.data_dir / "version-1").mkdir()
            return "version-1"

        monkeypatch.setitem(sys.modules, "shardcast", SimpleNamespace(broadcast=broadcast))
        await relay.sync_current()
        await relay.sync_current()

        index = json.loads(relay.index_path.read_bytes())
        assert index["policies"][manifest.policy_id] == {
            "artifact_digest": artifact.digest,
            "policy_digest": policy_manifest_digest(manifest),
            "shard_count": 1,
            "shard_size_bytes": config.shard_size_bytes,
            "size_bytes": artifact.size_bytes,
            "version": "version-1",
        }

    assert broadcasts == [artifact_bytes]
    external = [request for request in requests if request.url.host == "cdn.test"]
    assert len(external) == 1
    assert "authorization" not in external[0].headers
    await client.aclose()
