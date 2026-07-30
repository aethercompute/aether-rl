from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

from aether_rl.configs.worker import WorkerConfig
from aether_rl.protocol import PolicyManifest
from aether_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS, cleanup_process

from .client import CoordinatorClient, CoordinatorProtocolError
from .policy_cache import AdapterCache, CachedPolicy
from .spool import WorkerState


class WorkerVLLMSupervisor:
    def __init__(self, config: WorkerConfig, state: WorkerState):
        self.config = config
        self.state = state
        self.base_url = f"http://127.0.0.1:{config.inference_port}"
        self.process: asyncio.subprocess.Process | None = None
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=httpx.Timeout(10))
        self.config_path = state.root / "inference.toml"
        self.log_path = state.root / "inference.log"

    async def start(self, stop_event: asyncio.Event) -> None:
        if self.process is not None:
            raise RuntimeError("worker inference process is already started")
        self._write_config()
        log = open(self.log_path, "ab", buffering=0)
        environment = os.environ.copy()
        environment.update(DEFAULT_COMMON_ENV_VARS)
        environment.update(DEFAULT_INFERENCE_ENV_VARS)
        if self.config.hf_cache_dir is not None:
            environment["HF_HOME"] = str(self.config.hf_cache_dir)
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "aether_rl.entrypoints.inference",
                "@",
                str(self.config_path),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                env=environment,
            )
        finally:
            log.close()
        deadline = asyncio.get_running_loop().time() + self.config.inference_startup_timeout_seconds
        while True:
            if stop_event.is_set():
                raise asyncio.CancelledError
            if self.process.returncode is not None:
                raise RuntimeError(
                    f"worker inference process exited during startup with code {self.process.returncode}"
                )
            if await self._ready():
                return
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("worker inference process did not become ready")
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=0.5)
            except TimeoutError:
                pass

    async def stop(self) -> None:
        process = self.process
        self.process = None
        if process is not None and process.returncode is None:
            await asyncio.to_thread(cleanup_process, process.pid, signal.SIGTERM)
            try:
                await asyncio.wait_for(process.wait(), timeout=self.config.inference_shutdown_timeout_seconds)
            except TimeoutError:
                await asyncio.to_thread(cleanup_process, process.pid, signal.SIGKILL)
                await process.wait()
        await self.client.aclose()

    async def monitor(self) -> None:
        if self.process is None:
            raise RuntimeError("worker inference process is not started")
        return_code = await self.process.wait()
        raise RuntimeError(f"worker inference process exited unexpectedly with code {return_code}")

    async def _ready(self) -> bool:
        try:
            health, liveness, models = await asyncio.gather(
                self.client.get("/health"),
                self.client.get("/liveness"),
                self.client.get("/v1/models"),
            )
            if health.status_code != 200 or liveness.status_code != 200 or models.status_code != 200:
                return False
            model_ids = {item["id"] for item in models.json()["data"]}
            return self.config.base_model.model_name in model_ids
        except (httpx.HTTPError, KeyError, TypeError, ValueError):
            return False

    def _write_config(self) -> None:
        def quoted(value: str) -> str:
            return json.dumps(value)

        lines = [
            "enable_lora = true",
            f"max_loras = {self.config.max_loaded_policies}",
            f"max_cpu_loras = {self.config.max_loaded_policies}",
            f"max_lora_rank = {self.config.max_lora_rank}",
            f"gpu_memory_utilization = {self.config.gpu_memory_utilization}",
            "",
            "[server]",
            'host = "127.0.0.1"',
            f"port = {self.config.inference_port}",
            "",
            "[model]",
            f"name = {quoted(self.config.base_model.model_name)}",
            f"revision = {quoted(self.config.base_model.model_revision)}",
            f"trust_remote_code = {str(self.config.trust_remote_code).lower()}",
            *([f"max_model_len = {self.config.max_model_len}"] if self.config.max_model_len is not None else []),
            "",
            "[tokenizer]",
            f"name = {quoted(self.config.base_model.tokenizer_name)}",
            f"revision = {quoted(self.config.base_model.tokenizer_revision)}",
            "",
            "[parallel]",
            f"tp = {self.config.tensor_parallel_size}",
            "",
        ]
        data = "\n".join(lines).encode()
        temporary = self.config_path.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.config_path)
        self.state._fsync_directory(self.config_path.parent)


class VLLMAdminClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def models(self) -> dict[str, str | None]:
        response = await self.client.get("/v1/models")
        response.raise_for_status()
        return {item["id"]: item.get("root") for item in response.json()["data"]}

    async def load(self, name: str, path: Path) -> None:
        models = await self.models()
        if name in models:
            if Path(models[name]).resolve() != path.resolve() if models[name] is not None else True:
                raise CoordinatorProtocolError("loaded policy name points to different adapter bytes")
            return
        try:
            response = await self.client.post(
                "/v1/load_lora_adapter",
                json={"lora_name": name, "lora_path": str(path)},
            )
            response.raise_for_status()
            models = await self.models()
            if name not in models or models[name] is None or Path(models[name]).resolve() != path.resolve():
                raise CoordinatorProtocolError("vLLM did not load the exact immutable adapter path")
        except Exception as error:
            raise PolicyRuntimeFatalError("vLLM adapter load state is ambiguous") from error

    async def unload(self, name: str) -> None:
        try:
            response = await self.client.post("/v1/unload_lora_adapter", json={"lora_name": name})
            response.raise_for_status()
            if name in await self.models():
                raise CoordinatorProtocolError("vLLM still reports an unloaded adapter")
        except Exception as error:
            raise PolicyRuntimeFatalError("vLLM adapter unload state is ambiguous") from error


class PolicyRuntimeFatalError(RuntimeError):
    worker_fatal = True


class WorkerPolicyRuntime:
    def __init__(
        self,
        config: WorkerConfig,
        state: WorkerState,
        coordinator: CoordinatorClient,
        *,
        supervisor: WorkerVLLMSupervisor | None = None,
    ):
        self.config = config
        self.supervisor = supervisor or WorkerVLLMSupervisor(config, state)
        self.admin = VLLMAdminClient(self.supervisor.client)
        self.cache = AdapterCache(state, coordinator, max_bytes=config.adapter_cache_max_bytes)
        self._loaded: dict[str, CachedPolicy | None] = {}
        self._references: dict[str, int] = {}
        self._last_used: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)

    async def start(self, stop_event: asyncio.Event) -> None:
        await self.supervisor.start(stop_event)

    async def stop(self) -> None:
        await self.supervisor.stop()
        self._loaded.clear()

    async def monitor(self) -> None:
        await self.supervisor.monitor()

    def loaded_policy_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaded))

    @asynccontextmanager
    async def acquire(self, manifest: PolicyManifest):
        async with self.cache.pin(manifest) as cached:
            async with self._condition:
                if cached is not None and manifest.policy_id not in self._loaded:
                    await self._wait_for_load_capacity()
                    await self.admin.load(manifest.served_model_name, cached.path)
                    self.cache.mark_loaded(manifest.policy_id)
                self._loaded[manifest.policy_id] = cached
                self._references[manifest.policy_id] = self._references.get(manifest.policy_id, 0) + 1
                self._last_used[manifest.policy_id] = time.monotonic()
                await self._enforce_loaded_retention()
            try:
                yield manifest.served_model_name
            finally:
                async with self._condition:
                    remaining = self._references[manifest.policy_id] - 1
                    if remaining:
                        self._references[manifest.policy_id] = remaining
                    else:
                        self._references.pop(manifest.policy_id, None)
                    self._last_used[manifest.policy_id] = time.monotonic()
                    await self._enforce_loaded_retention()
                    self._condition.notify_all()

    async def _wait_for_load_capacity(self) -> None:
        while (
            len([cached for cached in self._loaded.values() if cached is not None]) >= self.config.max_loaded_policies
        ):
            candidates = [
                policy_id
                for policy_id, cached in self._loaded.items()
                if cached is not None and not self._references.get(policy_id, 0)
            ]
            if not candidates:
                await self._condition.wait()
                continue
            policy_id = min(candidates, key=lambda item: (self._last_used.get(item, 0), item))
            await self.admin.unload(policy_id)
            self.cache.mark_unloaded(policy_id)
            self._loaded.pop(policy_id, None)

    async def _enforce_loaded_retention(self) -> None:
        trained = [policy_id for policy_id, cached in self._loaded.items() if cached is not None]
        while len(trained) > self.config.max_loaded_policies:
            candidates = [policy_id for policy_id in trained if not self._references.get(policy_id, 0)]
            if not candidates:
                return
            policy_id = min(candidates, key=lambda item: (self._last_used.get(item, 0), item))
            await self.admin.unload(policy_id)
            self.cache.mark_unloaded(policy_id)
            self._loaded.pop(policy_id, None)
            trained.remove(policy_id)
