from pathlib import Path

from pydantic import Field, HttpUrl, model_validator

from aether_rl.utils.config import BaseConfig


class WorkerBaseModelIdentityConfig(BaseConfig):
    model_name: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    model_config_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    tokenizer_name: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    chat_template_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    vocab_size: int = Field(ge=1)
    quantization: str = Field(default="none", min_length=1)


class WorkerEnvironmentConfig(BaseConfig):
    id: str = Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@+~-]*$")
    package: str = Field(min_length=1, max_length=255)
    revision: str = Field(min_length=1, max_length=255)


class WorkerConfig(BaseConfig):
    coordinator_url: HttpUrl
    state_dir: Path = Path("worker-state")
    base_model: WorkerBaseModelIdentityConfig
    environments: list[WorkerEnvironmentConfig] = Field(min_length=1)
    execution_slots: int = Field(default=1, ge=1)
    tensor_parallel_size: int = Field(default=1, ge=1)
    labels: dict[str, str] = Field(default_factory=dict)
    spool_max_entries: int = Field(default=1000, ge=1)
    heartbeat_interval_seconds: float = Field(default=10, gt=0, allow_inf_nan=False)
    lease_wait_seconds: float = Field(default=30, ge=0, le=60, allow_inf_nan=False)
    request_timeout_seconds: float = Field(default=45, gt=0, allow_inf_nan=False)
    retry_min_seconds: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    retry_max_seconds: float = Field(default=30, gt=0, allow_inf_nan=False)
    shutdown_grace_seconds: float = Field(default=30, ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_worker(self) -> "WorkerConfig":
        identities = [(environment.id, environment.revision) for environment in self.environments]
        if identities != sorted(set(identities)):
            raise ValueError("environments must be sorted and unique")
        if self.retry_max_seconds < self.retry_min_seconds:
            raise ValueError("retry_max_seconds must be at least retry_min_seconds")
        if self.spool_max_entries < self.execution_slots:
            raise ValueError("spool_max_entries must be at least execution_slots")
        if any(not key or not value for key, value in self.labels.items()):
            raise ValueError("worker labels must have non-empty keys and values")
        if self.coordinator_url.path not in {None, "", "/"} or self.coordinator_url.query is not None:
            raise ValueError("coordinator_url must contain only scheme and authority")
        if self.coordinator_url.scheme != "https" and self.coordinator_url.host not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("coordinator_url must use HTTPS outside loopback")
        return self
