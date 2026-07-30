from pathlib import Path

from pydantic import Field, HttpUrl, model_validator

from aether_rl.utils.config import BaseConfig


class PolicyRelayConfig(BaseConfig):
    coordinator_url: HttpUrl
    state_dir: Path = Path("policy-relay-state")
    port: int = Field(default=8000, ge=1, le=65535)
    poll_interval_seconds: float = Field(default=2, gt=0, allow_inf_nan=False)
    request_timeout_seconds: float = Field(default=300, gt=0, allow_inf_nan=False)
    max_versions: int = Field(default=8, ge=2)
    shard_size_bytes: int = Field(default=8 * 1024**2, ge=1024**2)
    policy_download_allowed_origins: list[HttpUrl] = []

    @model_validator(mode="after")
    def validate_relay(self) -> "PolicyRelayConfig":
        if self.coordinator_url.path not in {None, "", "/"} or self.coordinator_url.query is not None:
            raise ValueError("coordinator_url must contain only scheme and authority")
        if self.coordinator_url.scheme != "https" and self.coordinator_url.host not in {
            "localhost",
            "127.0.0.1",
            "::1",
        }:
            raise ValueError("coordinator_url must use HTTPS outside loopback")
        for origin in self.policy_download_allowed_origins:
            if origin.scheme != "https":
                raise ValueError("policy download origins must use HTTPS")
            if origin.path not in {None, "", "/"} or origin.query is not None:
                raise ValueError("policy download origins must contain only scheme and authority")
        return self
