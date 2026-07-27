from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal, TypeAlias

import msgspec
from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_serializer, model_validator
from verifiers.utils.serve_utils import msgpack_encoder
from verifiers.v1.episode import WireEpisode
from verifiers.v1.types import SamplingConfig

PROTOCOL_VERSION = 1

ProtocolVersion: TypeAlias = Literal[1]
OpaqueId: TypeAlias = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=255,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@+~-]*$",
    ),
]
Digest: TypeAlias = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]
FullRevision: TypeAlias = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
Timestamp: TypeAlias = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ProtocolModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BaseModelIdentity(ProtocolModel):
    model_name: str = Field(min_length=1, max_length=512)
    model_revision: FullRevision
    model_config_digest: Digest
    tokenizer_name: str = Field(min_length=1, max_length=512)
    tokenizer_revision: FullRevision
    tokenizer_digest: Digest
    chat_template_digest: Digest
    vocab_size: int = Field(ge=1)
    quantization: str = Field(default="none", min_length=1, max_length=128)


class RuntimeIdentity(ProtocolModel):
    aether_rl_version: str = Field(min_length=1, max_length=64)
    python_version: str = Field(min_length=1, max_length=64)
    torch_version: str = Field(min_length=1, max_length=64)
    transformers_version: str = Field(min_length=1, max_length=64)
    vllm_version: str = Field(min_length=1, max_length=64)
    cuda_version: str | None = Field(default=None, min_length=1, max_length=64)


class EnvironmentIdentity(ProtocolModel):
    id: OpaqueId
    revision: str = Field(min_length=1, max_length=255)


class AdapterFile(ProtocolModel):
    name: Literal["adapter_config.json", "adapter_model.safetensors"]
    size_bytes: int = Field(gt=0)
    digest: Digest


class AdapterManifest(ProtocolModel):
    manifest_version: Literal[1] = 1
    digest: Digest
    files: tuple[AdapterFile, ...]
    rank: int = Field(ge=1)
    alpha: float = Field(gt=0, allow_inf_nan=False)
    target_modules: tuple[str, ...]

    @model_validator(mode="after")
    def validate_manifest(self) -> AdapterManifest:
        names = tuple(file.name for file in self.files)
        expected_names = ("adapter_config.json", "adapter_model.safetensors")
        if names != expected_names:
            raise ValueError(f"adapter files must be ordered exactly as {expected_names}")
        if not self.target_modules:
            raise ValueError("target_modules must not be empty")
        if any(not target for target in self.target_modules):
            raise ValueError("target_modules must contain non-empty names")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ValueError("target_modules must not contain duplicates")
        if tuple(sorted(self.target_modules)) != self.target_modules:
            raise ValueError("target_modules must be sorted")
        if self.digest != _adapter_manifest_digest(self):
            raise ValueError("adapter manifest digest does not match its contents")
        return self


class PolicyManifest(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    manifest_version: Literal[1] = 1
    run_id: OpaqueId
    policy_version: int = Field(ge=0)
    base_model: BaseModelIdentity
    adapter: AdapterManifest | None = None
    created_at: Timestamp
    policy_id: OpaqueId | None = None
    served_model_name: str | None = Field(default=None, min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_policy_version(self) -> PolicyManifest:
        if self.policy_version == 0 and self.adapter is not None:
            raise ValueError("base policy version 0 must not have an adapter")
        if self.policy_version > 0 and self.adapter is None:
            raise ValueError("trained policy versions must include an adapter")
        identity_digest = _policy_identity_digest(self)
        expected_policy_id = f"policy-v{self.policy_version:08d}-{identity_digest.removeprefix('sha256:')[:16]}"
        expected_model_name = self.base_model.model_name if self.adapter is None else expected_policy_id
        if self.policy_id is not None and self.policy_id != expected_policy_id:
            raise ValueError("policy_id does not match the policy version and adapter")
        if self.served_model_name is not None and self.served_model_name != expected_model_name:
            raise ValueError("served_model_name does not match the policy contents")
        object.__setattr__(self, "policy_id", expected_policy_id)
        object.__setattr__(self, "served_model_name", expected_model_name)
        return self


class WorkerCapabilities(ProtocolModel):
    base_model: BaseModelIdentity
    runtime: RuntimeIdentity
    environments: tuple[EnvironmentIdentity, ...] = Field(min_length=1)
    max_concurrent_assignments: int = Field(ge=1)
    gpu_count: int = Field(ge=1)
    tensor_parallel_size: int = Field(ge=1)
    labels: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_capabilities(self) -> WorkerCapabilities:
        environment_keys = tuple((environment.id, environment.revision) for environment in self.environments)
        if len(set(environment_keys)) != len(environment_keys):
            raise ValueError("environments must not contain duplicates")
        if tuple(sorted(environment_keys)) != environment_keys:
            raise ValueError("environments must be sorted by id and revision")
        if self.tensor_parallel_size > self.gpu_count:
            raise ValueError("tensor_parallel_size must not exceed gpu_count")
        return self


class WorkerRegistration(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    worker_id: OpaqueId
    worker_session_id: OpaqueId
    registered_at: Timestamp
    capabilities: WorkerCapabilities


class WorkerRegistrationResponse(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    worker_id: OpaqueId
    worker_session_id: OpaqueId
    created: bool
    server_time: Timestamp


class RolloutAssignment(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    assignment_id: OpaqueId
    group_id: OpaqueId
    group_index: int = Field(ge=0)
    group_size: int = Field(ge=1)
    kind: Literal["train", "eval"]
    environment: EnvironmentIdentity
    task_data: dict[str, JsonValue]
    sampling: SamplingConfig
    policy: PolicyManifest
    created_at: Timestamp
    deadline_at: Timestamp | None = None
    result_size_limit_bytes: int = Field(default=64 * 1024 * 1024, ge=1)

    @model_validator(mode="after")
    def validate_assignment(self) -> RolloutAssignment:
        if self.group_index >= self.group_size:
            raise ValueError("group_index must be less than group_size")
        if self.deadline_at is not None and self.deadline_at <= self.created_at:
            raise ValueError("deadline_at must be later than created_at")
        canonical_json_bytes(self.task_data)
        canonical_json_bytes(self.sampling)
        return self

    @model_serializer(mode="wrap")
    def serialize_json_safe(self, handler: Callable[[RolloutAssignment], dict[str, object]]) -> dict[str, object]:
        payload = handler(self)
        canonical_json_bytes(payload)
        return payload


class AssignmentLease(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    lease_id: OpaqueId
    attempt: int = Field(ge=1)
    worker_id: OpaqueId
    worker_session_id: OpaqueId
    issued_at: Timestamp
    expires_at: Timestamp
    assignment: RolloutAssignment

    @model_validator(mode="after")
    def validate_expiry(self) -> AssignmentLease:
        if self.expires_at <= self.issued_at:
            raise ValueError("expires_at must be later than issued_at")
        if self.issued_at < self.assignment.created_at:
            raise ValueError("issued_at must not be earlier than assignment.created_at")
        if self.assignment.deadline_at is not None and self.expires_at > self.assignment.deadline_at:
            raise ValueError("expires_at must not exceed assignment.deadline_at")
        return self


class WorkerHeartbeat(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    worker_id: OpaqueId
    worker_session_id: OpaqueId
    sent_at: Timestamp
    active_lease_ids: tuple[OpaqueId, ...] = ()
    loaded_policy_ids: tuple[OpaqueId, ...] = ()

    @model_validator(mode="after")
    def validate_unique_ids(self) -> WorkerHeartbeat:
        _validate_sorted_unique(self.active_lease_ids, "active_lease_ids")
        _validate_sorted_unique(self.loaded_policy_ids, "loaded_policy_ids")
        return self


class LeaseRequest(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    request_id: OpaqueId
    worker_id: OpaqueId
    worker_session_id: OpaqueId
    sent_at: Timestamp
    loaded_policy_ids: tuple[OpaqueId, ...] = ()
    environments: tuple[EnvironmentIdentity, ...] = Field(min_length=1)
    available_slots: int = Field(ge=1)
    wait_seconds: float = Field(default=0, ge=0, le=60, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_worker_state(self) -> LeaseRequest:
        _validate_sorted_unique(self.loaded_policy_ids, "loaded_policy_ids")
        environment_keys = tuple((environment.id, environment.revision) for environment in self.environments)
        _validate_sorted_unique(environment_keys, "environments")
        return self


class LeaseRenewRequest(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    assignment_id: OpaqueId
    lease_id: OpaqueId
    worker_id: OpaqueId
    worker_session_id: OpaqueId
    sent_at: Timestamp


class LeaseRenewal(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    assignment_id: OpaqueId
    lease_id: OpaqueId
    expires_at: Timestamp


class WorkerHeartbeatResponse(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    server_time: Timestamp
    renewals: tuple[LeaseRenewal, ...] = ()
    stop_lease_ids: tuple[OpaqueId, ...] = ()

    @model_validator(mode="after")
    def validate_response_ids(self) -> WorkerHeartbeatResponse:
        _validate_sorted_unique(tuple(renewal.lease_id for renewal in self.renewals), "renewal lease IDs")
        _validate_sorted_unique(self.stop_lease_ids, "stop_lease_ids")
        return self


class LeaseRenewResponse(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    server_time: Timestamp
    action: Literal["renewed", "stop"]
    renewal: LeaseRenewal | None = None
    reason: Literal["policy_stale", "cancelled"] | None = None

    @model_validator(mode="after")
    def validate_action(self) -> LeaseRenewResponse:
        if self.action == "renewed" and (self.renewal is None or self.reason is not None):
            raise ValueError("renewed action requires renewal details and no stop reason")
        if self.action == "stop" and (self.renewal is not None or self.reason is None):
            raise ValueError("stop action requires a reason and no renewal details")
        return self


class SubmissionResponse(ProtocolModel):
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    assignment_id: OpaqueId
    envelope_digest: Digest
    duplicate: bool
    terminal: bool


class ResultEnvelope(ProtocolModel):
    type: Literal["result"] = "result"
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    assignment_id: OpaqueId
    attempt: int = Field(ge=1)
    lease_id: OpaqueId
    worker_id: OpaqueId
    worker_session_id: OpaqueId
    requested_policy_id: OpaqueId
    served_policy_id: OpaqueId
    requested_policy_digest: Digest
    served_policy_digest: Digest
    completed_at: Timestamp
    result_digest: Digest
    episode: WireEpisode

    @model_validator(mode="after")
    def validate_result(self) -> ResultEnvelope:
        if self.requested_policy_id != self.served_policy_id:
            raise ValueError("served policy ID does not match the requested policy ID")
        if self.requested_policy_digest != self.served_policy_digest:
            raise ValueError("served policy digest does not match the requested policy digest")
        if self.result_digest != episode_digest(self.episode):
            raise ValueError("result_digest does not match the episode payload")
        return self

    @model_serializer(mode="wrap")
    def serialize_verified(self, handler: Callable[[ResultEnvelope], dict[str, object]]) -> dict[str, object]:
        if self.result_digest != episode_digest(self.episode):
            raise ValueError("result_digest does not match the episode payload")
        return handler(self)


class FailureEnvelope(ProtocolModel):
    type: Literal["failure"] = "failure"
    protocol_version: ProtocolVersion = PROTOCOL_VERSION
    assignment_id: OpaqueId
    attempt: int = Field(ge=1)
    lease_id: OpaqueId
    worker_id: OpaqueId
    worker_session_id: OpaqueId
    failed_at: Timestamp
    code: OpaqueId
    message: str = Field(min_length=1, max_length=8192)
    retryable: bool
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_details(self) -> FailureEnvelope:
        canonical_json_bytes(self.details)
        return self

    @model_serializer(mode="wrap")
    def serialize_json_safe(self, handler: Callable[[FailureEnvelope], dict[str, object]]) -> dict[str, object]:
        payload = handler(self)
        canonical_json_bytes(payload)
        return payload


TerminalEnvelope: TypeAlias = Annotated[ResultEnvelope | FailureEnvelope, Field(discriminator="type")]


def _validate_sorted_unique(values: tuple[object, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicates")
    if tuple(sorted(values)) != values:
        raise ValueError(f"{name} must be sorted")


def canonical_json_bytes(value: BaseModel | JsonValue) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def sha256_digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def episode_digest(episode: WireEpisode) -> str:
    payload = msgspec.msgpack.encode(
        episode.model_dump(mode="python"),
        enc_hook=msgpack_encoder,
        order="deterministic",
    )
    return sha256_digest(payload)


def result_envelope_bytes(envelope: ResultEnvelope) -> bytes:
    """Encode a complete result deterministically without losing binary trace sidecars."""
    payload = _sorted_mappings(envelope.model_dump(mode="python"))
    return msgspec.msgpack.encode(payload, enc_hook=msgpack_encoder)


def decode_result_envelope(payload: bytes) -> ResultEnvelope:
    return ResultEnvelope.model_validate(msgspec.msgpack.decode(payload))


def _sorted_mappings(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _sorted_mappings(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_mappings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_sorted_mappings(item) for item in value)
    return value


def _adapter_manifest_content(manifest: AdapterManifest) -> dict[str, JsonValue]:
    return manifest.model_dump(mode="json", exclude={"digest"})


def _adapter_manifest_digest(manifest: AdapterManifest) -> str:
    return sha256_digest(canonical_json_bytes(_adapter_manifest_content(manifest)))


def create_adapter_manifest(
    *,
    files: tuple[AdapterFile, ...],
    rank: int,
    alpha: float,
    target_modules: tuple[str, ...],
) -> AdapterManifest:
    data = {
        "manifest_version": 1,
        "files": [file.model_dump(mode="json") for file in files],
        "rank": rank,
        "alpha": float(alpha),
        "target_modules": list(target_modules),
    }
    return AdapterManifest(
        digest=sha256_digest(canonical_json_bytes(data)),
        files=files,
        rank=rank,
        alpha=alpha,
        target_modules=target_modules,
    )


def _policy_identity_digest(manifest: PolicyManifest) -> str:
    semantic_identity = {
        "protocol_version": manifest.protocol_version,
        "manifest_version": manifest.manifest_version,
        "run_id": manifest.run_id,
        "policy_version": manifest.policy_version,
        "base_model": manifest.base_model.model_dump(mode="json"),
        "adapter_digest": manifest.adapter.digest if manifest.adapter is not None else None,
    }
    return sha256_digest(canonical_json_bytes(semantic_identity))


def policy_manifest_digest(manifest: PolicyManifest) -> str:
    return _policy_identity_digest(manifest)
