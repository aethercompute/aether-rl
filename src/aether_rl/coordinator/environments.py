from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from verifiers.v1.types import SamplingConfig

from aether_rl.protocol import EnvironmentIdentity, OpaqueId, canonical_json_bytes


class EnvironmentSourceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: OpaqueId
    kind: Literal["train", "eval"]
    environment: EnvironmentIdentity
    tasks: tuple[dict[str, JsonValue], ...] = Field(min_length=1)
    sampling: SamplingConfig
    group_size: int = Field(ge=1)
    max_attempts: int = Field(ge=1)
    result_size_limit_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    assignment_timeout_seconds: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    weight: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    enabled: bool = True

    @model_validator(mode="after")
    def validate_canonical_payloads(self) -> EnvironmentSourceSpec:
        if not math.isfinite(self.weight):
            raise ValueError("weight must be finite")
        canonical_json_bytes(list(self.tasks))
        canonical_json_bytes(self.sampling)
        return self


class EnvironmentCatalog(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    sources: tuple[EnvironmentSourceSpec, ...]

    @model_validator(mode="after")
    def validate_source_ids(self) -> EnvironmentCatalog:
        source_ids = tuple(source.source_id for source in self.sources)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("source IDs must not contain duplicates")
        return self


def verifier_v1_task_payloads(tasks: Iterable[object]) -> tuple[dict[str, JsonValue], ...]:
    """Convert already-loaded verifier v1 tasks into central JSON payloads."""
    payloads = tuple(task.data.model_dump(mode="json") for task in tasks)  # type: ignore[attr-defined]
    canonical_json_bytes(list(payloads))
    if not payloads:
        raise ValueError("tasks must not be empty")
    return payloads
