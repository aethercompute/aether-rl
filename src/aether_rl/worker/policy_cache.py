from __future__ import annotations

import asyncio
import hashlib
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

from aether_rl.protocol import PolicyManifest, canonical_json_bytes, policy_manifest_digest, sha256_digest
from aether_rl.trainer.policy import POLICY_MANIFEST_NAME, verify_lora_policy

from .client import CoordinatorAPIError, CoordinatorClient, CoordinatorProtocolError
from .policy_transport import PolicyFileTransport
from .state import WorkerState


class AdapterCacheError(RuntimeError):
    pass


class AdapterCacheCorruptionError(AdapterCacheError):
    pass


@dataclass(frozen=True)
class CachedPolicy:
    manifest: PolicyManifest
    path: Path
    size_bytes: int


class AdapterCache:
    def __init__(
        self,
        state: WorkerState,
        client: CoordinatorClient,
        *,
        max_bytes: int,
        transport: PolicyFileTransport | None = None,
    ):
        self.state = state
        self.client = client
        self.max_bytes = max_bytes
        self.transport = transport
        self.root = state.root / "cache" / "policies"
        self.incoming = self.root / "incoming"
        self.artifacts = self.root / "sha256"
        for directory in (state.root / "cache", self.root, self.incoming, self.artifacts):
            state._create_directory(directory)
        self._cached: dict[Path, CachedPolicy] = {}
        self._clean_incoming()
        self._locks: dict[str, asyncio.Lock] = {}
        self._capacity_lock = asyncio.Lock()
        self._references: dict[str, int] = {}
        self._loaded: set[str] = set()
        self._last_used: dict[str, float] = {}
        self._cached = {entry.path: entry for entry in self._scan_entries()}
        self.enforce_retention()

    async def ensure(self, manifest: PolicyManifest) -> CachedPolicy | None:
        if manifest.adapter is None:
            return None
        adapter_size = sum(artifact.size_bytes for artifact in manifest.adapter.files)
        if adapter_size > self.max_bytes:
            raise AdapterCacheError("policy adapter exceeds adapter_cache_max_bytes")
        digest = policy_manifest_digest(manifest)
        lock = self._locks.setdefault(digest, asyncio.Lock())
        async with lock:
            async with self._capacity_lock:
                return await self._ensure_locked(manifest, adapter_size)

    @asynccontextmanager
    async def pin(self, manifest: PolicyManifest):
        cached = await self.ensure(manifest)
        policy_id = manifest.policy_id
        self._references[policy_id] = self._references.get(policy_id, 0) + 1
        self._last_used[policy_id] = time.monotonic()
        try:
            yield cached
        finally:
            remaining = self._references[policy_id] - 1
            if remaining:
                self._references[policy_id] = remaining
            else:
                self._references.pop(policy_id, None)
            self._last_used[policy_id] = time.monotonic()
            async with self._capacity_lock:
                await asyncio.to_thread(self.enforce_retention)

    def mark_loaded(self, policy_id: str) -> None:
        self._loaded.add(policy_id)
        self._last_used[policy_id] = time.monotonic()

    def mark_unloaded(self, policy_id: str) -> None:
        self._loaded.discard(policy_id)

    def enforce_retention(self) -> None:
        entries = self._entries()
        total = sum(entry.size_bytes for entry in entries)
        if total <= self.max_bytes:
            return
        for entry in sorted(
            entries, key=lambda item: (self._last_used.get(item.manifest.policy_id, 0), item.path.name)
        ):
            policy_id = entry.manifest.policy_id
            if policy_id in self._loaded or self._references.get(policy_id, 0):
                continue
            self._remove_entry(entry.path)
            total -= entry.size_bytes
            if total <= self.max_bytes:
                break
        self.state._fsync_directory(self.artifacts)

    async def _ensure_locked(self, manifest: PolicyManifest, adapter_size: int) -> CachedPolicy:
        final_path = self._path(manifest)
        if final_path.is_symlink():
            raise AdapterCacheCorruptionError("cached policy directory must not be a symlink")
        if final_path.exists():
            cached = self._cached.get(final_path)
            if cached is None:
                cached = await asyncio.to_thread(self._verify, manifest, final_path)
                self._cached[final_path] = cached
            self._last_used[manifest.policy_id] = time.monotonic()
            return cached
        partial_bytes = await asyncio.to_thread(self._prune_partial_files, manifest)
        await asyncio.to_thread(self._make_space, adapter_size - partial_bytes)
        await self._validate_remote_manifest(manifest)
        temporary = self.incoming / f".{manifest.policy_id}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir(mode=0o700)
        try:
            for artifact in manifest.adapter.files:
                await self._download_file(manifest, artifact.name, artifact.size_bytes, artifact.digest, temporary)
            manifest_path = temporary / POLICY_MANIFEST_NAME
            self._write_file(manifest_path, canonical_json_bytes(manifest))
            self.state._fsync_directory(temporary)
            await asyncio.to_thread(verify_lora_policy, temporary, expected=manifest)
            self._publish_directory(temporary, final_path)
            cached = await asyncio.to_thread(self._verify, manifest, final_path)
            self._cached[final_path] = cached
        finally:
            if temporary.exists() or temporary.is_symlink():
                self._remove_entry(temporary)
        self._last_used[manifest.policy_id] = time.monotonic()
        return cached

    def _make_space(self, required_bytes: int) -> None:
        entries = self._entries()
        total = sum(entry.size_bytes for entry in entries) + self._incoming_size()
        for entry in sorted(
            entries, key=lambda item: (self._last_used.get(item.manifest.policy_id, 0), item.path.name)
        ):
            if total + required_bytes <= self.max_bytes:
                break
            policy_id = entry.manifest.policy_id
            if policy_id in self._loaded or self._references.get(policy_id, 0):
                continue
            self._remove_entry(entry.path)
            total -= entry.size_bytes
        if total + required_bytes > self.max_bytes:
            raise AdapterCacheError("adapter cache has no evictable capacity for the assigned policy")
        self.state._fsync_directory(self.artifacts)

    def _publish_directory(self, temporary: Path, final_path: Path) -> None:
        digest_name = final_path.parent.name
        artifacts_fd = os.open(self.artifacts, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            try:
                os.mkdir(digest_name, mode=0o700, dir_fd=artifacts_fd)
            except FileExistsError:
                pass
            digest_fd = os.open(
                digest_name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=artifacts_fd,
            )
            incoming_fd = os.open(self.incoming, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                try:
                    os.rename(
                        temporary.name,
                        final_path.name,
                        src_dir_fd=incoming_fd,
                        dst_dir_fd=digest_fd,
                    )
                except FileExistsError:
                    if final_path.is_symlink() or not final_path.exists():
                        raise
                os.fsync(digest_fd)
                os.fsync(artifacts_fd)
            finally:
                os.close(incoming_fd)
                os.close(digest_fd)
        finally:
            os.close(artifacts_fd)

    async def _validate_remote_manifest(self, expected: PolicyManifest) -> None:
        response = await self.client.get_policy_manifest(expected.policy_id)
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise CoordinatorProtocolError("policy manifest uses unsupported content encoding")
        if response.headers.get("content-type", "").split(";", 1)[0].lower() != "application/json":
            raise CoordinatorProtocolError("policy manifest has the wrong content type")
        if len(response.content) > 1024 * 1024:
            raise CoordinatorProtocolError("policy manifest exceeds its size limit")
        if response.headers.get("etag") != f'"{sha256_digest(response.content)}"':
            raise CoordinatorProtocolError("policy manifest ETag does not match its bytes")
        try:
            manifest = PolicyManifest.model_validate_json(response.content)
        except ValueError as error:
            raise CoordinatorProtocolError("policy manifest is malformed") from error
        if canonical_json_bytes(manifest) != response.content or manifest != expected:
            raise CoordinatorProtocolError("policy manifest does not match the assignment")

    async def _download_file(
        self,
        manifest: PolicyManifest,
        name: str,
        size_bytes: int,
        digest: str,
        directory: Path,
    ) -> None:
        part_path = self.incoming / f"{digest.removeprefix('sha256:')}.{name}.part"
        if part_path.is_symlink() or (part_path.exists() and not part_path.is_file()):
            raise AdapterCacheCorruptionError("partial policy file path is unsafe")
        if part_path.exists() and part_path.stat().st_size > size_bytes:
            part_path.unlink()

        if self.transport is not None and not part_path.exists():
            downloaded = await self.transport.download_shardcast(
                manifest,
                name,
                size_bytes=size_bytes,
                digest=digest,
                destination=part_path,
            )
            if downloaded:
                received, hasher = self._partial_digest(part_path)
                if received != size_bytes or f"sha256:{hasher.hexdigest()}" != digest:
                    part_path.unlink(missing_ok=True)

        attempts = self.transport.attempts if self.transport is not None else 1
        for attempt in range(attempts):
            offset, hasher = self._partial_digest(part_path)
            if offset == size_bytes:
                if f"sha256:{hasher.hexdigest()}" == digest:
                    break
                part_path.unlink()
                offset, hasher = 0, hashlib.sha256()
            if offset == 0 and part_path.exists():
                part_path.unlink()
            try:
                if self.transport is None:
                    stream = self.client.stream_policy_file(manifest.policy_id, name, offset=offset)
                    require_digest_etag = True
                else:
                    stream = self.transport.stream(
                        manifest,
                        name,
                        offset=offset,
                        coordinator_only=(
                            self.transport.coordinator_fallback and attempts > 1 and attempt == attempts - 1
                        ),
                    )
                    require_digest_etag = None
                async with stream as streamed:
                    if self.transport is None:
                        response = streamed
                    else:
                        response = streamed.response
                        require_digest_etag = streamed.require_digest_etag
                    self._validate_response(
                        response,
                        name=name,
                        size_bytes=size_bytes,
                        digest=digest,
                        offset=offset,
                        require_digest_etag=bool(require_digest_etag),
                    )
                    flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
                    flags |= os.O_APPEND if offset else os.O_EXCL
                    descriptor = os.open(part_path, flags, 0o600)
                    with os.fdopen(descriptor, "ab" if offset else "wb") as file:
                        async for chunk in response.aiter_bytes():
                            offset += len(chunk)
                            if offset > size_bytes:
                                raise CoordinatorProtocolError("policy file exceeds its declared size")
                            file.write(chunk)
                            hasher.update(chunk)
                        file.flush()
                        os.fsync(file.fileno())
                if offset != size_bytes or f"sha256:{hasher.hexdigest()}" != digest:
                    part_path.unlink(missing_ok=True)
                    raise CoordinatorProtocolError("policy file bytes do not match its manifest")
            except CoordinatorAPIError as error:
                if not error.retryable or attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(error.retry_after or min(2**attempt, 5))
            except (httpx.TransportError, httpx.TimeoutException, CoordinatorProtocolError):
                if attempt + 1 >= attempts:
                    raise
                await asyncio.sleep(min(2**attempt, 5))

        received, hasher = self._partial_digest(part_path)
        if received != size_bytes or f"sha256:{hasher.hexdigest()}" != digest:
            part_path.unlink(missing_ok=True)
            raise CoordinatorProtocolError("policy file bytes do not match its manifest")
        path = directory / name
        os.replace(part_path, path)
        self.state._fsync_directory(directory)

    @staticmethod
    def _partial_digest(path: Path) -> tuple[int, object]:
        hasher = hashlib.sha256()
        received = 0
        if path.exists():
            with open(path, "rb") as file:
                while chunk := file.read(1024 * 1024):
                    received += len(chunk)
                    hasher.update(chunk)
        return received, hasher

    @staticmethod
    def _validate_response(
        response,
        *,
        name: str,
        size_bytes: int,
        digest: str,
        offset: int,
        require_digest_etag: bool,
    ) -> None:
        if response.headers.get("content-encoding", "identity").lower() != "identity":
            raise CoordinatorProtocolError("policy file uses unsupported content encoding")
        expected_type = "application/json" if name == "adapter_config.json" else "application/octet-stream"
        if response.headers.get("content-type", "").split(";", 1)[0].lower() != expected_type:
            raise CoordinatorProtocolError("policy file has the wrong content type")
        if require_digest_etag and response.headers.get("etag") != f'"{digest}"':
            raise CoordinatorProtocolError("policy file ETag does not match its manifest")
        try:
            content_length = int(response.headers.get("content-length", ""))
        except ValueError as error:
            raise CoordinatorProtocolError("policy file Content-Length is invalid") from error
        if content_length != size_bytes - offset:
            raise CoordinatorProtocolError("policy file Content-Length does not match its manifest")
        content_range = response.headers.get("content-range")
        expected_range = None if not offset else f"bytes {offset}-{size_bytes - 1}/{size_bytes}"
        if content_range != expected_range:
            raise CoordinatorProtocolError("policy file Content-Range does not match the requested offset")

    def _verify(self, manifest: PolicyManifest, path: Path) -> CachedPolicy:
        if path.is_symlink():
            raise AdapterCacheCorruptionError(f"cached policy {manifest.policy_id} is a symlink")
        try:
            verified = verify_lora_policy(path, expected=manifest)
        except Exception as error:
            raise AdapterCacheCorruptionError(f"cached policy {manifest.policy_id} is corrupt") from error
        size = sum(artifact.size_bytes for artifact in verified.adapter.files) if verified.adapter else 0
        return CachedPolicy(verified, path, size)

    def _scan_entries(self) -> list[CachedPolicy]:
        entries: list[CachedPolicy] = []
        for digest_dir in self.artifacts.iterdir():
            if digest_dir.is_symlink() or not digest_dir.is_dir():
                raise AdapterCacheCorruptionError("adapter cache contains an unsafe digest entry")
            for policy_dir in digest_dir.iterdir():
                if policy_dir.is_symlink():
                    raise AdapterCacheCorruptionError("adapter cache contains a symlinked policy")
                try:
                    manifest = PolicyManifest.model_validate_json((policy_dir / POLICY_MANIFEST_NAME).read_bytes())
                    if self._path(manifest) != policy_dir:
                        raise AdapterCacheCorruptionError("cached policy is stored under the wrong content address")
                    entries.append(self._verify(manifest, policy_dir))
                except Exception as error:
                    if isinstance(error, AdapterCacheCorruptionError):
                        raise
                    raise AdapterCacheCorruptionError("adapter cache contains an invalid policy") from error
        return entries

    def _entries(self) -> list[CachedPolicy]:
        return list(self._cached.values())

    def _path(self, manifest: PolicyManifest) -> Path:
        digest = policy_manifest_digest(manifest).removeprefix("sha256:")
        return self.artifacts / digest / manifest.policy_id

    def _clean_incoming(self) -> None:
        for path in self.incoming.iterdir():
            if (
                path.is_file()
                and not path.is_symlink()
                and re.fullmatch(
                    r"[0-9a-f]{64}\.(adapter_config\.json|adapter_model\.safetensors)\.part",
                    path.name,
                )
            ):
                continue
            self._remove_entry(path)
        self.state._fsync_directory(self.incoming)

    def _prune_partial_files(self, manifest: PolicyManifest) -> int:
        expected = {
            f"{artifact.digest.removeprefix('sha256:')}.{artifact.name}.part": artifact.size_bytes
            for artifact in manifest.adapter.files
        }
        total = 0
        for path in self.incoming.iterdir():
            if path.name not in expected or path.is_symlink() or not path.is_file():
                self._remove_entry(path)
            elif path.stat().st_size > expected[path.name]:
                path.unlink()
            else:
                total += path.stat().st_size
        self.state._fsync_directory(self.incoming)
        return total

    def _incoming_size(self) -> int:
        return sum(path.stat().st_size for path in self.incoming.iterdir() if path.is_file() and not path.is_symlink())

    @staticmethod
    def _write_file(path: Path, data: bytes) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())

    def _remove_entry(self, path: Path) -> None:
        self._cached.pop(path, None)
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        else:
            raise AdapterCacheCorruptionError(f"cannot remove unsafe cache entry: {path}")
