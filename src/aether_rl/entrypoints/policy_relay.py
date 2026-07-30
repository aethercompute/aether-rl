from __future__ import annotations

import asyncio
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from aether_rl.configs.policy_relay import PolicyRelayConfig
from aether_rl.protocol import PolicyManifest, canonical_json_bytes, policy_manifest_digest
from aether_rl.utils.config import cli
from aether_rl.utils.process import set_proc_title
from aether_rl.worker.client import CoordinatorAPIError, CoordinatorClient, CoordinatorProtocolError
from aether_rl.worker.spool import WorkerState


class ShardcastPolicyRelay:
    def __init__(self, config: PolicyRelayConfig, coordinator: CoordinatorClient, state: WorkerState):
        self.config = config
        self.coordinator = coordinator
        self.state = state
        self.data_dir = state.root / "shardcast"
        self.download_dir = state.root / "downloads"
        state._create_directory(self.data_dir)
        state._create_directory(self.download_dir)
        self.index_path = self.data_dir / "aether-policies.json"
        self.allowed_origins = {self._origin(str(origin)) for origin in config.policy_download_allowed_origins}

    async def run(self) -> None:
        import shardcast

        shardcast.initialize(
            str(self.data_dir),
            port=self.config.port,
            max_distribution_folders=self.config.max_versions,
        )
        try:
            while True:
                try:
                    await self.sync_current()
                except (httpx.HTTPError, CoordinatorAPIError, CoordinatorProtocolError, OSError, ValueError) as error:
                    print(f"Policy relay sync failed: {error}", flush=True)
                await asyncio.sleep(self.config.poll_interval_seconds)
        finally:
            shardcast.shutdown()

    async def sync_current(self) -> None:
        manifest = await self.coordinator.get_current_policy()
        if manifest.adapter is None:
            return
        index = self._load_index()
        digest = policy_manifest_digest(manifest)
        existing = index["policies"].get(manifest.policy_id)
        if existing is not None and existing.get("policy_digest") == digest:
            version_dir = self.data_dir / existing["version"]
            if version_dir.is_dir():
                return

        artifact = next(file for file in manifest.adapter.files if file.name == "adapter_model.safetensors")
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{manifest.policy_id}.", suffix=".tmp", dir=self.download_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            await self._download(manifest, artifact.name, artifact.size_bytes, artifact.digest, temporary)
            import shardcast

            version = await asyncio.to_thread(shardcast.broadcast, str(temporary), self.config.shard_size_bytes)
            shard_count = (artifact.size_bytes + self.config.shard_size_bytes - 1) // self.config.shard_size_bytes
            index["policies"][manifest.policy_id] = {
                "policy_digest": digest,
                "artifact_digest": artifact.digest,
                "size_bytes": artifact.size_bytes,
                "version": version,
                "shard_count": shard_count,
                "shard_size_bytes": self.config.shard_size_bytes,
            }
            self._publish_index(index)
        finally:
            temporary.unlink(missing_ok=True)

    async def _download(self, manifest: PolicyManifest, name: str, size: int, digest: str, path: Path) -> None:
        response = None
        if self.allowed_origins:
            locations = await self.coordinator.get_policy_locations(manifest.policy_id)
            if locations.policy_digest != policy_manifest_digest(manifest):
                raise CoordinatorProtocolError("policy locations do not match the active policy")
            location = next(file for file in locations.files if file.name == name)
            if self._origin(location.url) not in self.allowed_origins:
                raise CoordinatorProtocolError("policy location uses an unapproved origin")
            request = self.coordinator.artifact_client.build_request(
                "GET", location.url, headers={"Accept-Encoding": "identity"}
            )
            response = await self.coordinator.artifact_client.send(request, stream=True)
            if response.status_code != 200:
                await response.aclose()
                response = None
        if response is None:
            context = self.coordinator.stream_policy_file(manifest.policy_id, name)
            response = await context.__aenter__()
        else:
            context = None
        try:
            hasher = hashlib.sha256()
            received = 0
            with open(path, "wb") as file:
                async for chunk in response.aiter_bytes():
                    received += len(chunk)
                    if received > size:
                        raise CoordinatorProtocolError("policy file exceeds its manifest size")
                    file.write(chunk)
                    hasher.update(chunk)
                file.flush()
                os.fsync(file.fileno())
            if received != size or f"sha256:{hasher.hexdigest()}" != digest:
                raise CoordinatorProtocolError("policy file does not match its manifest")
        finally:
            if context is None:
                await response.aclose()
            else:
                await context.__aexit__(None, None, None)

    def _load_index(self) -> dict:
        if not self.index_path.exists():
            return {"version": 1, "policies": {}}
        data = json.loads(self.index_path.read_bytes())
        if data.get("version") != 1 or not isinstance(data.get("policies"), dict):
            raise ValueError("SHARDCAST policy index is malformed")
        return data

    def _publish_index(self, index: dict) -> None:
        data = canonical_json_bytes(index)
        temporary = self.data_dir / f".{self.index_path.name}.tmp"
        with open(temporary, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.index_path)
        self.state._fsync_directory(self.data_dir)

    @staticmethod
    def _origin(url: str) -> tuple[str, str]:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc:
            raise CoordinatorProtocolError("policy relay locations must use HTTPS")
        return parsed.scheme, parsed.netloc.lower()


async def run_relay(config: PolicyRelayConfig) -> None:
    token = os.environ.get("AETHER_COORDINATOR_TOKEN")
    if not token:
        raise RuntimeError("AETHER_COORDINATOR_TOKEN is required")
    state = WorkerState(config.state_dir)
    coordinator = CoordinatorClient(
        str(config.coordinator_url),
        token,
        timeout_seconds=config.request_timeout_seconds,
    )
    try:
        await ShardcastPolicyRelay(config, coordinator, state).run()
    finally:
        await coordinator.close()
        state.close()


def main() -> None:
    set_proc_title("Policy Relay")
    asyncio.run(run_relay(cli(PolicyRelayConfig)))


if __name__ == "__main__":
    main()
