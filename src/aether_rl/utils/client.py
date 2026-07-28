from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from itertools import cycle
from typing import Protocol, runtime_checkable

import httpx
import verifiers.v1 as vf
from httpx import AsyncClient
from openai import AsyncOpenAI
from renderers import RendererConfig
from verifiers.v1.clients.config import EvalClientConfig, TrainClientConfig

from aether_rl.configs.shared import ClientConfig

ClientIdentity = tuple[str, str | None]


def client_identity(client: vf.ClientConfig) -> ClientIdentity:
    return (client.base_url, client.headers.get("X-data-parallel-rank"))


@runtime_checkable
class InferencePool(Protocol):
    model_name: str

    @property
    def train_clients(self) -> list[vf.ClientConfig]: ...

    async def get_eval_client(self) -> vf.ClientConfig: ...

    async def select_train_client(self, load: Mapping[ClientIdentity, int]) -> vf.ClientConfig: ...

    async def wait_for_ready(self, model_name: str, timeout: int | None = None) -> None: ...

    async def score(self, token_ids: list[int]) -> list[float]: ...

    async def stop(self) -> None: ...


class PrefillScorer:
    def __init__(self) -> None:
        self._clients: dict[ClientIdentity, AsyncOpenAI] = {}
        self._round_robin = 0

    async def score(self, configs: list[vf.ClientConfig], model: str, token_ids: list[int]) -> list[float]:
        if not configs:
            raise RuntimeError("no inference endpoints available to prefill-score")
        config = configs[self._round_robin % len(configs)]
        self._round_robin += 1
        identity = client_identity(config)
        client = self._clients.get(identity)
        if client is None:
            client = self._clients[identity] = AsyncOpenAI(
                base_url=config.base_url,
                api_key=os.environ.get(config.api_key_var) or "EMPTY",
                default_headers=config.headers or None,
            )
        return await prefill_logprobs(client, model, token_ids)

    async def close(self) -> None:
        await asyncio.gather(*(client.close() for client in self._clients.values()))


class StaticInferencePool:
    def __init__(
        self,
        client_config: ClientConfig,
        model_name: str,
        train_client_type: str = "openai_chat_completions",
        eval_client_type: str = "openai_chat_completions",
        renderer_config: RendererConfig | None = None,
        pool_size: int | None = None,
    ):
        self._train_clients = setup_clients(
            client_config,
            client_type=train_client_type,
            renderer_config=renderer_config,
            renderer_model_name=model_name if train_client_type == "renderer" else None,
            pool_size=pool_size,
        )
        self._eval_clients = setup_clients(client_config, client_type=eval_client_type)
        self._admin_clients = setup_admin_clients(client_config)
        self._skip_model_check = client_config.skip_model_check
        self._wait_for_ready_timeout = client_config.wait_for_ready_timeout
        self._eval_cycle = cycle(self._eval_clients)
        self._scorer = PrefillScorer()
        self.model_name = model_name

    @property
    def train_clients(self) -> list[vf.ClientConfig]:
        return self._train_clients

    async def get_eval_client(self) -> vf.ClientConfig:
        return next(self._eval_cycle)

    async def select_train_client(self, load: Mapping[ClientIdentity, int]) -> vf.ClientConfig:
        return min(self.train_clients, key=lambda client: load[client_identity(client)])

    async def wait_for_ready(self, model_name: str, timeout: int | None = None) -> None:
        await check_health(self._admin_clients, timeout=timeout or self._wait_for_ready_timeout)
        await maybe_check_has_model(self._admin_clients, model_name, self._skip_model_check)

    async def score(self, token_ids: list[int]) -> list[float]:
        return await self._scorer.score(self.train_clients, self.model_name, token_ids)

    async def stop(self) -> None:
        await self._scorer.close()
        await asyncio.gather(*(client.aclose() for client in self._admin_clients))


async def setup_inference_pool(
    client_config: ClientConfig,
    model_name: str,
    train_client_type: str = "openai_chat_completions",
    eval_client_type: str = "openai_chat_completions",
    renderer_config: RendererConfig | None = None,
    pool_size: int | None = None,
) -> InferencePool:
    return StaticInferencePool(
        client_config,
        model_name,
        train_client_type,
        eval_client_type,
        renderer_config,
        pool_size,
    )


def setup_clients(
    client_config: ClientConfig,
    client_type: str = "openai_chat_completions",
    renderer_config: RendererConfig | None = None,
    renderer_model_name: str | None = None,
    pool_size: int | None = None,
) -> list[vf.ClientConfig]:
    config_class = TrainClientConfig if client_type == "renderer" else EvalClientConfig
    environment_headers = {
        name: value
        for name, value in ((name, os.getenv(variable)) for name, variable in client_config.headers_from_env.items())
        if value is not None
    }
    renderer_fields = (
        {"renderer": renderer_config, "pool_size": pool_size or 1, "renderer_model_name": renderer_model_name}
        if client_type == "renderer"
        else {}
    )
    clients = []
    for base_url in client_config.base_url:
        for rank in range(client_config.dp_rank_count):
            headers = {**client_config.headers, **environment_headers}
            if client_config.dp_rank_count > 1:
                headers["X-data-parallel-rank"] = str(rank)
            clients.append(
                config_class(
                    base_url=base_url,
                    api_key_var=client_config.api_key_var,
                    headers=headers,
                    **renderer_fields,
                )
            )
    return clients


def setup_admin_clients(client_config: ClientConfig) -> list[AsyncClient]:
    headers = {
        name: value
        for name, value in ((name, os.getenv(variable)) for name, variable in client_config.headers_from_env.items())
        if value is not None
    }
    headers.update(client_config.headers)
    api_key = os.getenv(client_config.api_key_var)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return [
        AsyncClient(base_url=url.rstrip("/").removesuffix("/v1"), headers=headers, timeout=httpx.Timeout(None))
        for url in client_config.base_url
    ]


async def maybe_check_has_model(clients: list[AsyncClient], model_name: str, skip_model_check: bool = False) -> None:
    if skip_model_check:
        return
    for client, response in zip(clients, await asyncio.gather(*(client.get("/v1/models") for client in clients))):
        response.raise_for_status()
        if not any(model["id"] == model_name for model in response.json()["data"]):
            raise ValueError(f"Model {model_name} was not found on {client.base_url}")


async def check_health(clients: list[AsyncClient], interval: int = 1, timeout: int = 1800) -> None:
    async def wait(client: AsyncClient) -> None:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            try:
                response = await client.get("/health")
                response.raise_for_status()
                return
            except httpx.HTTPError:
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(f"Inference server is not ready: {client.base_url}") from None
                await asyncio.sleep(interval)

    await asyncio.gather(*(wait(client) for client in clients))


async def prefill_logprobs(openai: AsyncOpenAI, model: str, token_ids: list[int]) -> list[float]:
    from vllm.entrypoints.serve.disagg.protocol import GenerateResponse

    base = str(openai.base_url).rstrip("/").removesuffix("/v1")
    response = await openai.post(
        f"{base}/inference/v1/generate",
        cast_to=httpx.Response,
        body={
            "model": model,
            "token_ids": token_ids,
            "sampling_params": {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "prompt_logprobs": 1},
        },
    )
    parsed = GenerateResponse.model_validate_json(response.content)
    logprobs = []
    for entry in parsed.prompt_logprobs or []:
        if not entry:
            logprobs.append(0.0)
            continue
        first = next(iter(entry.values()))
        value = first.logprob if hasattr(first, "logprob") else first.get("logprob")
        logprobs.append(float(value) if value is not None else 0.0)
    return logprobs
