from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from aether_rl.configs.worker import WorkerConfig
from aether_rl.protocol import PolicyManifest, policy_manifest_digest

from .client import CoordinatorAPIError, CoordinatorClient, CoordinatorProtocolError


@dataclass(frozen=True)
class PolicyFileResponse:
    response: httpx.Response
    require_digest_etag: bool


class PolicyFileTransport:
    def __init__(self, config: WorkerConfig, coordinator: CoordinatorClient, *, clock=time.time):
        self.coordinator = coordinator
        self.clock = clock
        self.attempts = config.policy_download_attempts
        self.coordinator_fallback = config.policy_coordinator_fallback
        self.allowed_origins = {self._origin(str(origin)) for origin in config.policy_download_allowed_origins}
        self.shardcast_servers = [str(server).rstrip("/") for server in config.shardcast_servers]
        self.shardcast_download_concurrency = config.shardcast_download_concurrency

    async def download_shardcast(
        self,
        manifest: PolicyManifest,
        name: str,
        *,
        size_bytes: int,
        digest: str,
        destination: Path,
    ) -> bool:
        if name != "adapter_model.safetensors" or not self.shardcast_servers:
            return False
        record = await self._shardcast_record(manifest)
        if record is None:
            return False
        expected = {
            "policy_digest": policy_manifest_digest(manifest),
            "artifact_digest": digest,
            "size_bytes": size_bytes,
        }
        if any(record.get(key) != value for key, value in expected.items()):
            return False
        version = record.get("version")
        shard_count = record.get("shard_count")
        shard_size = record.get("shard_size_bytes")
        if (
            not isinstance(version, str)
            or not version.startswith("v")
            or type(shard_count) is not int
            or type(shard_size) is not int
            or shard_size < 1024**2
            or shard_count != (size_bytes + shard_size - 1) // shard_size
        ):
            return False

        temporary = Path(tempfile.mkdtemp(prefix=".shardcast-", dir=destination.parent))
        semaphore = asyncio.Semaphore(self.shardcast_download_concurrency)
        try:

            async def download(index: int) -> Path:
                async with semaphore:
                    filename = f"shard_{index + 1:05d}.bin"
                    path = temporary / filename
                    expected_size = min(shard_size, size_bytes - index * shard_size)
                    for server in self.shardcast_servers:
                        try:
                            request = self.coordinator.artifact_client.build_request(
                                "GET", f"{server}/{version}/{filename}", headers={"Accept-Encoding": "identity"}
                            )
                            response = await self.coordinator.artifact_client.send(request, stream=True)
                        except (httpx.TransportError, httpx.TimeoutException):
                            continue
                        try:
                            if (
                                response.status_code != 200
                                or response.headers.get("content-encoding", "identity").lower() != "identity"
                            ):
                                continue
                            try:
                                content_length = int(response.headers.get("content-length", ""))
                            except ValueError:
                                continue
                            if content_length != expected_size:
                                continue
                            received = 0
                            with open(path, "wb") as file:
                                async for chunk in response.aiter_bytes():
                                    received += len(chunk)
                                    if received > expected_size:
                                        break
                                    file.write(chunk)
                                file.flush()
                                os.fsync(file.fileno())
                            if received == expected_size:
                                return path
                            path.unlink(missing_ok=True)
                        finally:
                            await response.aclose()
                    raise CoordinatorProtocolError(f"SHARDCAST could not download {filename}")

            try:
                shards = []
                for start in range(0, shard_count, self.shardcast_download_concurrency):
                    stop = min(start + self.shardcast_download_concurrency, shard_count)
                    shards.extend(await asyncio.gather(*(download(index) for index in range(start, stop))))
            except (httpx.TransportError, httpx.TimeoutException, CoordinatorProtocolError):
                return False
            with open(destination, "wb") as output:
                for path in shards:
                    with open(path, "rb") as shard:
                        while chunk := shard.read(1024 * 1024):
                            output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            return destination.stat().st_size == size_bytes
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    async def _shardcast_record(self, manifest: PolicyManifest) -> dict | None:
        for server in self.shardcast_servers:
            try:
                response = await self.coordinator.artifact_client.get(
                    f"{server}/aether-policies.json", headers={"Accept-Encoding": "identity"}
                )
            except (httpx.TransportError, httpx.TimeoutException):
                continue
            if response.status_code != 200 or len(response.content) > 1024 * 1024:
                continue
            try:
                index = json.loads(response.content)
                policies = index["policies"] if index["version"] == 1 else {}
                record = policies.get(manifest.policy_id)
            except (KeyError, TypeError, ValueError):
                continue
            if isinstance(record, dict):
                return record
        return None

    @asynccontextmanager
    async def stream(self, manifest: PolicyManifest, name: str, *, offset: int = 0, coordinator_only: bool = False):
        if self.allowed_origins and not coordinator_only:
            try:
                external_url = await self._external_url(manifest, name)
                response = None if external_url is None else await self._open_external(external_url, offset=offset)
            except (httpx.TransportError, httpx.TimeoutException, CoordinatorProtocolError):
                if not self.coordinator_fallback:
                    raise
            else:
                if response is not None:
                    try:
                        yield PolicyFileResponse(response, require_digest_etag=False)
                    finally:
                        await response.aclose()
                    return
        if not self.coordinator_fallback:
            raise CoordinatorProtocolError("no usable external policy source is available")
        async with self.coordinator.stream_policy_file(manifest.policy_id, name, offset=offset) as response:
            yield PolicyFileResponse(response, require_digest_etag=True)

    async def _external_url(self, manifest: PolicyManifest, name: str) -> str | None:
        try:
            locations = await self.coordinator.get_policy_locations(manifest.policy_id)
        except CoordinatorAPIError as error:
            if error.status_code == 404 and self.coordinator_fallback:
                return None
            raise
        if locations.policy_id != manifest.policy_id or locations.policy_digest != policy_manifest_digest(manifest):
            raise CoordinatorProtocolError("policy locations do not match the assigned policy")
        if locations.expires_at <= self.clock():
            raise CoordinatorProtocolError("policy locations expired before download")
        urls = {file.name: file.url for file in locations.files}
        url = urls.get(name)
        if url is None or self._origin(url) not in self.allowed_origins:
            raise CoordinatorProtocolError("policy location uses an unapproved origin")
        return url

    async def _open_external(self, url: str, *, offset: int) -> httpx.Response:
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = self.coordinator.artifact_client.build_request("GET", url, headers=headers)
        response = await self.coordinator.artifact_client.send(request, stream=True)
        expected = 206 if offset else 200
        if response.status_code != expected:
            await response.aread()
            status = response.status_code
            await response.aclose()
            raise CoordinatorProtocolError(f"external policy source returned HTTP {status}")
        return response

    @staticmethod
    def _origin(url: str) -> tuple[str, str]:
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None or parsed.password is not None:
            raise CoordinatorProtocolError("policy location must use an HTTPS origin without user information")
        return parsed.scheme, parsed.netloc.lower()
