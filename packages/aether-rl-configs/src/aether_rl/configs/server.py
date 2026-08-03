import time
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, HttpUrl, model_validator
from verifiers.v1.types import SamplingConfig

from aether_rl.configs.algorithm import AlgoConfig, GRPOAlgoConfig
from aether_rl.configs.filters import FilterConfig
from aether_rl.utils.config import BaseConfig


class ServerBaseModelIdentityConfig(BaseConfig):
    model_name: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    tokenizer_name: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    chat_template_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    vocab_size: int = Field(ge=1)
    quantization: str = Field(default="none", min_length=1)


class ServerSourceConfig(BaseConfig):
    source_id: str = Field(min_length=1)
    kind: Literal["train", "eval"] = "train"
    environment_id: str = Field(min_length=1)
    environment_revision: str = Field(min_length=1)
    environment_config: dict[str, Any]
    sampling: SamplingConfig = SamplingConfig(temperature=1.0, max_tokens=1024)
    group_size: int = Field(default=8, ge=1)
    max_attempts: int = Field(default=3, ge=1)
    task_limit: int | None = Field(default=None, ge=1)
    shuffle_seed: int | None = None
    result_size_limit_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    assignment_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    enabled: bool = True
    processing_id: str = Field(default="v1", min_length=1)
    algorithm: AlgoConfig = GRPOAlgoConfig()
    pre_filters: list[FilterConfig] = []
    post_filters: list[FilterConfig] = []


class S3PolicyDistributionConfig(BaseConfig):
    type: Literal["s3"] = "s3"
    bucket: str = Field(min_length=1)
    prefix: str = Field(default="aether-policies", min_length=1)
    endpoint_url: HttpUrl | None = None
    region: str = Field(default="auto", min_length=1)
    presign_ttl_seconds: int = Field(default=900, ge=60, le=604800)

    @model_validator(mode="after")
    def validate_s3(self) -> "S3PolicyDistributionConfig":
        if self.prefix.startswith("/") or self.prefix.endswith("/") or "//" in self.prefix:
            raise ValueError("policy distribution prefix must not have leading, trailing, or repeated slashes")
        if self.endpoint_url is not None:
            if self.endpoint_url.host is None or self.endpoint_url.host.startswith("."):
                raise ValueError("policy distribution endpoint must have a valid host")
            if self.endpoint_url.scheme != "https" and self.endpoint_url.host not in {
                "localhost",
                "127.0.0.1",
                "::1",
            }:
                raise ValueError("policy distribution endpoint must use HTTPS outside loopback")
        return self


class ServerConfig(BaseConfig):
    run_id: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@+~-]*$")
    run_root: Path = Path("server-state")
    database_path: Path | None = None
    base_model: ServerBaseModelIdentityConfig
    sources: list[ServerSourceConfig] = Field(min_length=1)
    trainer_config_path: Path
    trainer_output_dir: Path | None = None
    trainer_processes: int = Field(default=1, ge=1)
    training_batch_size: int = Field(default=1, ge=1)
    published_checkpoint_keep_last: int | None = Field(default=None, ge=1)
    """Retain this many active/recent full checkpoints after durable policy publication. None retains all checkpoints."""
    service_interval_seconds: float = Field(default=0.2, gt=0, allow_inf_nan=False)
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    dry_run: bool = False
    created_at: float = Field(default_factory=time.time, ge=0, allow_inf_nan=False)
    control_body_limit_bytes: int = Field(default=1024 * 1024, ge=1)
    inference_body_limit_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    environment_slots: int = Field(default=1, ge=1)
    lease_duration_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    loaded_policy_preference_seconds: float = Field(default=5, ge=0, allow_inf_nan=False)
    max_policy_lag: int = Field(default=0, ge=0)
    max_lease_wait_seconds: float = Field(default=30, ge=0, allow_inf_nan=False)
    durable_provider_timeout_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    lease_poll_interval_seconds: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    stale_after_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
    lease_reaper_interval_seconds: float = Field(default=1, gt=0, allow_inf_nan=False)
    policy_verification_interval_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    policy_distribution: S3PolicyDistributionConfig | None = None

    @model_validator(mode="after")
    def validate_server(self) -> "ServerConfig":
        if self.database_path is not None and self.database_path.is_dir():
            raise ValueError("database_path must be a SQLite file path, not a directory")
        if not self.trainer_config_path.is_file():
            raise ValueError("trainer_config_path must be an existing TOML file")
        source_ids = [source.source_id for source in self.sources]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("source IDs must be unique")
        if self.base_model.model_revision == "0" * 40 or self.base_model.tokenizer_revision == "0" * 40:
            raise ValueError("base model and tokenizer revisions must be real pinned commits, not placeholders")
        placeholder_digest = "sha256:" + "0" * 64
        if placeholder_digest in {
            self.base_model.model_config_digest,
            self.base_model.tokenizer_digest,
            self.base_model.chat_template_digest,
        }:
            raise ValueError("base model identity digests must be real fingerprints, not placeholders")
        return self
