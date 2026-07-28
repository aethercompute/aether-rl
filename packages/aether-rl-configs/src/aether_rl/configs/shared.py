import os
from typing import Annotated, Literal, TypeAlias

from pydantic import AfterValidator, Field

from aether_rl.utils.config import BaseConfig

# Launcher-managed env vars that a component's `env_vars` must not set: GPU partitioning
# and the single shared W&B run. The launcher always sets these last, so allowing them in
# `env_vars` would be a silent no-op (or, on multi-node, a footgun) — reject them instead.
PROTECTED_ENV_VARS = frozenset(
    {"CUDA_VISIBLE_DEVICES", "WANDB_SHARED_MODE", "WANDB_SHARED_RUN_ID", "WANDB_SHARED_LABEL"}
)


def reject_protected_env_vars(env_vars: dict[str, str]) -> dict[str, str]:
    clobbered = sorted(PROTECTED_ENV_VARS & env_vars.keys())
    if clobbered:
        raise ValueError(
            f"env_vars cannot set launcher-managed vars {clobbered} — set by the launcher, not overridable"
        )
    return env_vars


EnvVars: TypeAlias = Annotated[dict[str, str], AfterValidator(reject_protected_env_vars)]
"""A per-component `env_vars` mapping, validated to not clobber `PROTECTED_ENV_VARS`."""


ServerType = Literal["vllm", "openai"]


class VLMConfig(BaseConfig):
    vision_encoder_attr: str
    """Dotted attribute path to the vision encoder module (e.g. ``model.visual``)."""

    language_model_attr: str
    """Dotted attribute path to the language model module (e.g. ``model.language_model``)."""

    freeze_vision_encoder: bool = True
    """Freeze the vision encoder parameters during training."""


class BaseModelConfig(BaseConfig):
    name: str = "Qwen/Qwen3-0.6B"
    """HF model name or local path."""

    revision: str | None = None
    """Model revision. Distributed runs require a full 40-character Hugging Face commit SHA."""

    trust_remote_code: bool = False
    """Trust remote code when initializing the tokenizer."""

    vlm: "VLMConfig | None" = None
    """VLM configuration. Setting this enables vision-language model support."""


class TokenizerConfig(BaseConfig):
    name: str | None = None
    """Tokenizer name or path. If None, the model's default tokenizer is used."""

    revision: str | None = None
    """Tokenizer revision. If None, inherits the model revision when the tokenizer name is inherited."""

    trust_remote_code: bool | None = None
    """Trust remote code when initializing the tokenizer. If None, inherits the model setting."""

    chat_template: str | None = None
    """Jinja2 chat template string or template file path. If None, uses the tokenizer default."""


class ClientConfig(BaseConfig):
    wait_for_ready_timeout: int = 1800
    """Seconds to wait at startup for the inference pool to become ready. Applies to both the static health check and elastic DNS-based discovery."""

    base_url: list[str] = ["http://localhost:8000/v1"]
    """Base URLs for the OpenAI API. With more than one URL, the client round-robins (chat) completion requests across all servers. Ignored when ``elastic`` is set."""

    api_key_var: str = "VLLM_API_KEY"
    """Environment variable name containing the API key, resolved via ``os.getenv``. Can be any string when the server is not protected by an API key; the same key is used for every URL."""

    headers: dict[str, str] = {}
    """Static headers sent with every request."""

    headers_from_env: dict[str, str] = {}
    """Maps HTTP header names to environment variable names; each entry is resolved via ``os.getenv`` and merged into request headers. e.g. ``{"X-Prime-Team-ID": "PRIME_TEAM_ID"}``."""

    extra_headers_from_state: dict[str, str] = {}
    """Maps HTTP header names to rollout-state field names. The header value is read from the rollout state dict on every request. e.g. ``{"X-Session-ID": "trajectory_id"}`` enables sticky routing at the inference router."""

    skip_model_check: bool = False
    """Skip checking that the model is available in the inference pool. Useful for external APIs or keys that do not expose ``/models``."""

    dp_rank_count: int = Field(1, ge=1)
    """Number of data-parallel ranks behind each base URL. When > 1, each URL is expanded into ``dp_rank_count`` logical clients pinned via the ``X-data-parallel-rank`` header, so every request within a rollout hits the same DP engine and reuses KV cache. Auto-set from the inference config when using the RL entrypoint."""


class LogConfig(BaseConfig):
    level: str = Field(default_factory=lambda: os.environ.get("PRIME_LOG_LEVEL", "info"))
    """Log level for the process. Defaults to ``$PRIME_LOG_LEVEL`` if set, else ``info``."""

    vf_level: str = Field(default_factory=lambda: os.environ.get("PRIME_VF_LOG_LEVEL", "info"))
    """Log level for the verifiers package. Defaults to ``$PRIME_VF_LOG_LEVEL`` if set, else ``info``."""

    json_logging: bool = False
    """Emit newline-delimited JSON logs for aggregation (Loki, Grafana, etc.)."""

    log_data: bool = False
    """Log the first data sample at startup."""

    interval: float = Field(10.0, gt=0)
    """Interval (seconds) for periodic logs across components."""


class TrainerLogConfig(LogConfig):
    ranks_filter: list[int] = [0]
    """Trainer ranks to show in console output. Passed to ``torchrun --local-ranks-filter``."""


class LogExtrasConfig(BaseConfig):
    samples: bool = True
    """Log prompt/response samples."""

    distributions: bool = True
    """Log distributions (rewards, advantages, etc.)."""

    interval: int = Field(10, ge=1)
    """Step interval between extras logs."""

    sample_ratio: float | None = Field(None, ge=0.0, le=1.0)
    """Fraction of rollouts to log per step. The effective cap is ``len(rollouts) * sample_ratio``; 1.0 = all, 0.5 = half, 0.0 = none."""


class WandbConfig(BaseConfig):
    # Shared configs (May be overwritten by WandbConfig from `rl.py`)
    project: str = "aether-rl"
    """W&B project to log to."""

    entity: str | None = None
    """W&B entity to log to."""

    name: str | None = None
    """W&B run name."""

    group: str | None = None
    """W&B group."""

    tags: list[str] | None = None
    """W&B tags attached to the run."""

    offline: bool = False
    """Run W&B in offline mode."""

    log_metrics: list[str] | None = None
    """Exact metric names to send to W&B. None sends every metric."""

    create_overview: bool = True
    """Create a curated project workspace view when online."""


class WandbWithExtrasConfig(WandbConfig):
    log_extras: LogExtrasConfig | None = LogExtrasConfig()
    """Extras logging configuration. If None, no extras are logged."""


class FileMonitorConfig(BaseConfig):
    """Enable the local JSONL metric sink (``<output_dir>/metrics.jsonl``).

    Present (non-None) enables it, mirroring the ``prime_monitor`` pattern. Metrics
    are the same scalars sent to W&B; useful for self-hosted dashboards or when W&B
    is disabled.
    """

    filename: str = "metrics.jsonl"
    """Name of the JSONL file written under the component's ``output_dir``."""


class PrimeMonitorConfig(BaseConfig):
    base_url: str = "https://api.primeintellect.ai/api/v1/rft"
    """Base URL for the Prime Intellect monitoring API."""

    api_key_var: str = "PRIME_API_KEY"
    """Environment variable name containing the Prime Intellect API key, resolved via ``os.getenv``."""

    log_extras: LogExtrasConfig | None = LogExtrasConfig()
    """Extras logging configuration. If None, no extras are logged."""

    run_name: str | None = None
    """Run name shown on the platform. Defaults to the W&B run name when set, otherwise the platform auto-generates one."""

    team_id: str | None = None
    """Team ID to associate the run with."""

    frontend_url: str | None = None
    """Frontend base URL used for the dashboard link printed after registration. Defaults to the Prime CLI frontend URL when unset."""


class HeartbeatConfig(BaseConfig):
    url: str
    """URL to send the heartbeat to."""


class MetricsServerConfig(BaseConfig):
    port: int = Field(8000, ge=1, le=65535)
    """Port to expose metrics and health endpoints on."""

    host: str = "0.0.0.0"
    """Host to bind the server to."""


class BaseTransportConfig(BaseConfig):
    pass


class FileSystemTransportConfig(BaseTransportConfig):
    type: Literal["filesystem"] = "filesystem"


TransportConfig: TypeAlias = FileSystemTransportConfig
