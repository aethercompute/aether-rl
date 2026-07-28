from argparse import Namespace
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from pydantic_config import BaseConfig

from aether_rl.configs.shared import BaseModelConfig, EnvVars, LogConfig, TokenizerConfig
from aether_rl.utils.parsers import resolve_reasoning_parser, resolve_tool_call_parser


class ServerConfig(BaseConfig):
    host: str | None = None
    port: int = 8000


class ParallelConfig(BaseConfig):
    tp: int = Field(1, ge=1)
    dp: int = Field(1, ge=1)

    def __str__(self) -> str:
        return f"tp={self.tp} dp={self.dp}"


class ModelConfig(BaseModelConfig):
    dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    max_model_len: int | None = None
    enforce_eager: bool = False
    trust_remote_code: bool = False
    chat_template: str | None = None
    tool_call_parser: str | None = "auto"
    reasoning_parser: str | None = "auto"
    rope_scaling: dict[str, Any] | str | None = None

    @model_validator(mode="after")
    def auto_resolve_parsers(self):
        if self.tool_call_parser == "auto":
            self.tool_call_parser = resolve_tool_call_parser(self.name)
        if self.reasoning_parser == "auto":
            self.reasoning_parser = resolve_reasoning_parser(self.name)
        return self


class CPUOffloadTier(BaseConfig):
    num_bytes: int = Field(..., gt=0)


class DiskOffloadTier(BaseConfig):
    path: Path


class NativeKVCacheOffloadConfig(BaseConfig):
    type: Literal["native"] = "native"
    cpu: CPUOffloadTier
    disk: DiskOffloadTier | None = None

    def to_connector_dict(self) -> dict[str, Any]:
        extra: dict[str, Any] = {"cpu_bytes_to_use": int(self.cpu.num_bytes)}
        if self.disk is not None:
            extra["spec_name"] = "TieringOffloadingSpec"
            extra["secondary_tiers"] = [{"type": "fs_python", "root_dir": str(self.disk.path)}]
        return {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": extra,
        }


VALID_VLLM_LORA_RANKS = (8, 16, 32, 64, 128, 256, 320, 512)
All2AllBackend = Literal[
    "allgather_reducescatter",
    "deepep_high_throughput",
    "deepep_low_latency",
    "flashinfer_nvlink_one_sided",
    "flashinfer_nvlink_two_sided",
]


class InferenceConfig(BaseConfig):
    server: ServerConfig = ServerConfig()
    model: ModelConfig = Field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = Field(default_factory=TokenizerConfig)
    parallel: ParallelConfig = ParallelConfig()
    log: LogConfig = LogConfig()
    env_vars: EnvVars = {}
    enable_lora: bool = False
    max_loras: int = Field(8, ge=1)
    max_cpu_loras: int = Field(100, ge=1)
    max_lora_rank: int | None = None
    lora_target_modules: list[str] | None = None
    enable_prefix_caching: bool | None = None
    gpu_memory_utilization: float = Field(0.9, gt=0, le=1)
    quantization: str | None = None
    api_server_count: int = Field(1, ge=0)
    data_parallel_size_local: int | None = Field(None, ge=1)
    data_parallel_rpc_port: int = Field(13345, ge=1, le=65535)
    seed: int = 0
    enable_expert_parallel: bool = False
    all2all_backend: All2AllBackend = "allgather_reducescatter"
    enable_eplb: bool = False
    enable_dbo: bool = False
    use_deep_gemm: bool = False
    kv_cache_offload: NativeKVCacheOffloadConfig | None = None
    enable_return_routed_experts: bool = False
    enable_fp32_lm_head: bool = True
    enable_fp32_router_logits: bool = True
    vllm_extra: dict[str, Any] = {}
    dry_run: bool = False

    @model_validator(mode="after")
    def auto_setup_tokenizer(self):
        if self.tokenizer.name is None:
            self.tokenizer.name = self.model.name
            if self.tokenizer.revision is None:
                self.tokenizer.revision = self.model.revision
        if self.tokenizer.trust_remote_code is None:
            self.tokenizer.trust_remote_code = self.model.trust_remote_code
        return self

    @model_validator(mode="after")
    def auto_setup_kv_cache_offload(self):
        if self.kv_cache_offload is not None:
            if self.enable_prefix_caching is False:
                raise ValueError("KV cache offloading requires enable_prefix_caching")
            if "enable_prefix_caching" not in self.model_fields_set:
                self.enable_prefix_caching = True
        return self

    @model_validator(mode="after")
    def auto_setup_max_lora_rank(self):
        if self.max_lora_rank is None:
            return self
        original_rank = self.max_lora_rank
        for valid_rank in VALID_VLLM_LORA_RANKS:
            if valid_rank >= original_rank:
                self.max_lora_rank = valid_rank
                return self
        raise ValueError(f"max_lora_rank={original_rank} exceeds vLLM maximum of {VALID_VLLM_LORA_RANKS[-1]}")

    @model_validator(mode="after")
    def auto_setup_api_server_count(self):
        if self.vllm_extra.get("headless", False):
            self.api_server_count = 0
        elif self.enable_lora:
            self.api_server_count = 1
        elif "api_server_count" not in self.model_fields_set:
            self.api_server_count = max(self.api_server_count, self.data_parallel_size_local or self.parallel.dp)
        return self

    def to_vllm(self) -> Namespace:
        values = {
            "host": self.server.host,
            "port": self.server.port,
            "model": self.model.name,
            "revision": self.model.revision,
            "tokenizer": self.tokenizer.name,
            "tokenizer_revision": self.tokenizer.revision,
            "dtype": self.model.dtype,
            "max_model_len": self.model.max_model_len,
            "enforce_eager": self.model.enforce_eager,
            "trust_remote_code": self.model.trust_remote_code or bool(self.tokenizer.trust_remote_code),
            "chat_template": self.model.chat_template,
            "tool_call_parser": self.model.tool_call_parser,
            "reasoning_parser": self.model.reasoning_parser,
            "rope_scaling": self.model.rope_scaling,
            "tensor_parallel_size": self.parallel.tp,
            "data_parallel_size": self.parallel.dp,
            "data_parallel_size_local": self.data_parallel_size_local,
            "data_parallel_rpc_port": self.data_parallel_rpc_port,
            "enable_lora": self.enable_lora,
            "enable_prefix_caching": self.enable_prefix_caching,
            "max_loras": self.max_loras,
            "max_cpu_loras": self.max_cpu_loras,
            "max_lora_rank": self.max_lora_rank,
            "lora_target_modules": self.lora_target_modules,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "quantization": self.quantization,
            "api_server_count": self.api_server_count,
            "enable_return_routed_experts": self.enable_return_routed_experts,
            "enable_expert_parallel": self.enable_expert_parallel,
            "all2all_backend": self.all2all_backend,
            "enable_eplb": self.enable_eplb,
            "enable_dbo": self.enable_dbo,
            "seed": self.seed,
            "logprobs_mode": "processed_logprobs",
        }
        namespace = Namespace(**{key: value for key, value in values.items() if value is not None})
        namespace.enable_auto_tool_choice = hasattr(namespace, "tool_call_parser")
        if self.kv_cache_offload is not None:
            namespace.kv_transfer_config = self.kv_cache_offload.to_connector_dict()
        additional_config = {}
        if self.enable_fp32_lm_head:
            additional_config["fp32_lm_head"] = True
        if self.enable_fp32_router_logits:
            additional_config["fp32_router_logits"] = True
        if additional_config:
            namespace.additional_config = additional_config
        return namespace
