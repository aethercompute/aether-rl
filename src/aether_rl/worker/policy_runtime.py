from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
import tomli_w

from aether_rl.configs.worker import WorkerConfig
from aether_rl.protocol import PolicyManifest
from aether_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS, cleanup_process

from .client import CoordinatorClient, CoordinatorProtocolError
from .policy_cache import AdapterCache, CachedPolicy
from .policy_transport import PolicyFileTransport
from .state import WorkerState


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
        vllm_extra = dict(self.config.vllm_extra)
        if self.config.enable_chunked_prefill is not None:
            vllm_extra["enable_chunked_prefill"] = self.config.enable_chunked_prefill
        payload = {
            "enable_lora": True,
            "max_loras": self.config.max_loaded_policies,
            "max_cpu_loras": self.config.max_loaded_policies,
            "max_lora_rank": self.config.max_lora_rank,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "enable_dbo": self.config.enable_dbo,
            "server": {
                "host": "127.0.0.1",
                "port": self.config.inference_port,
            },
            "model": {
                "name": self.config.base_model.model_name,
                "revision": self.config.base_model.model_revision,
                "trust_remote_code": self.config.trust_remote_code,
            },
            "tokenizer": {
                "name": self.config.base_model.tokenizer_name,
                "revision": self.config.base_model.tokenizer_revision,
            },
            "parallel": {"tp": self.config.tensor_parallel_size},
        }
        if self.config.enable_prefix_caching is not None:
            payload["enable_prefix_caching"] = self.config.enable_prefix_caching
        if self.config.quantization is not None:
            payload["quantization"] = self.config.quantization
        if self.config.max_model_len is not None:
            payload["model"]["max_model_len"] = self.config.max_model_len
        if vllm_extra:
            payload["vllm_extra"] = vllm_extra
        data = tomli_w.dumps(payload).encode()
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

    async def inference_metrics(self) -> dict[str, float]:
        response = await self.client.get("/metrics")
        response.raise_for_status()
        return parse_vllm_metrics(response.text)


_PROMETHEUS_SAMPLE_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")


def parse_vllm_metrics(text: str) -> dict[str, float]:
    samples: dict[str, list[float]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROMETHEUS_SAMPLE_RE.match(line)
        if match is None:
            continue
        name, raw_value = match.groups()
        try:
            value = float(raw_value)
        except ValueError:
            continue
        samples.setdefault(name, []).append(value)

    def first(*names: str, aggregate=sum) -> float | None:
        for name in names:
            values = samples.get(name)
            if values:
                return float(aggregate(values))
        return None

    metrics: dict[str, float] = {}
    mappings = {
        "inference/agg/running_requests": ("vllm:num_requests_running", "vllm_num_requests_running"),
        "inference/agg/waiting_requests": ("vllm:num_requests_waiting", "vllm_num_requests_waiting"),
        "inference/agg/kv_cache_usage_mean": ("vllm:gpu_cache_usage_perc", "vllm_gpu_cache_usage_perc"),
        "inference/agg/prefix_cache_hit_rate": ("vllm:prefix_cache_hit_rate", "vllm_prefix_cache_hit_rate"),
    }
    for metric_name, sample_names in mappings.items():
        value = first(*sample_names, aggregate=lambda values: sum(values) / len(values))
        if value is not None:
            metrics[metric_name] = value
    return metrics


class PolicyRuntimeFatalError(RuntimeError):
    worker_fatal = True


logger = logging.getLogger(__name__)


@dataclass
class PolicyRuntimeMetrics:
    adapter_download_seconds: float = 0.0
    adapter_load_seconds: float = 0.0
    adapter_switch_seconds: float = 0.0
    adapter_acquires: int = 0

    def snapshot(self) -> dict[str, float]:
        acquires = max(self.adapter_acquires, 1)
        return {
            "inference/agg/adapter_download_time": self.adapter_download_seconds / acquires,
            "inference/agg/adapter_load_time": self.adapter_load_seconds / acquires,
            "inference/agg/adapter_switch_time": self.adapter_switch_seconds / acquires,
        }


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
        self.coordinator = coordinator
        self.cache = AdapterCache(
            state,
            coordinator,
            max_bytes=config.adapter_cache_max_bytes,
            transport=PolicyFileTransport(config, coordinator),
        )
        self._loaded: dict[str, CachedPolicy | None] = {}
        self._references: dict[str, int] = {}
        self._last_used: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._condition = asyncio.Condition(self._lock)
        self._stop_event: asyncio.Event | None = None
        self.metrics = PolicyRuntimeMetrics()

    async def start(self, stop_event: asyncio.Event) -> None:
        self._stop_event = stop_event
        await self.supervisor.start(stop_event)

    async def stop(self) -> None:
        await self.supervisor.stop()
        self._loaded.clear()

    async def monitor(self) -> None:
        if self.config.policy_prefetch_interval_seconds is None:
            await self.supervisor.monitor()
            return
        supervisor = asyncio.create_task(self.supervisor.monitor(), name="worker-vllm-monitor")
        prefetch = asyncio.create_task(self._prefetch_loop(), name="worker-policy-prefetch")
        try:
            done, _ = await asyncio.wait((supervisor, prefetch), return_when=asyncio.FIRST_COMPLETED)
            await next(iter(done))
        finally:
            supervisor.cancel()
            prefetch.cancel()
            await asyncio.gather(supervisor, prefetch, return_exceptions=True)

    def loaded_policy_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._loaded))

    async def metrics_snapshot(self) -> dict[str, float]:
        metrics = self.metrics.snapshot()
        try:
            metrics |= await self.admin.inference_metrics()
        except Exception as error:
            logger.debug("vLLM metrics scrape failed: %s", error)
        return metrics

    async def _prefetch_loop(self) -> None:
        if self._stop_event is None or self.config.policy_prefetch_interval_seconds is None:
            raise RuntimeError("policy prefetch started before the worker runtime")
        while not self._stop_event.is_set():
            try:
                manifest = await self.coordinator.get_current_policy()
                await self.cache.ensure(manifest)
            except Exception as error:
                logger.warning("policy prefetch failed: %s", error)
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self.config.policy_prefetch_interval_seconds,
                )
            except TimeoutError:
                pass

    @asynccontextmanager
    async def acquire(self, manifest: PolicyManifest):
        started_at = time.monotonic()
        async with self.cache.pin(manifest) as cached:
            self.metrics.adapter_download_seconds += time.monotonic() - started_at
            async with self._condition:
                if cached is not None and manifest.policy_id not in self._loaded:
                    await self._wait_for_load_capacity()
                    load_started_at = time.monotonic()
                    await self.admin.load(manifest.served_model_name, cached.path)
                    self.metrics.adapter_load_seconds += time.monotonic() - load_started_at
                    self.cache.mark_loaded(manifest.policy_id)
                self._loaded[manifest.policy_id] = cached
                self._references[manifest.policy_id] = self._references.get(manifest.policy_id, 0) + 1
                self._last_used[manifest.policy_id] = time.monotonic()
                await self._enforce_loaded_retention()
                self.metrics.adapter_switch_seconds += time.monotonic() - started_at
                self.metrics.adapter_acquires += 1
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
