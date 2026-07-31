import pytest
from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError
from verifiers.v1.episode import WireEpisode
from verifiers.v1.types import SamplingConfig

from aether_rl.protocol import (
    AdapterFile,
    AdapterManifest,
    AssignmentLease,
    BaseModelIdentity,
    EnvironmentIdentity,
    FailureEnvelope,
    PolicyManifest,
    ResultEnvelope,
    RolloutAssignment,
    TerminalEnvelope,
    WorkerCapabilities,
    canonical_json_bytes,
    create_adapter_manifest,
    episode_digest,
    policy_manifest_digest,
    sha256_digest,
)

REVISION = "a" * 40
DIGEST_A = "sha256:" + "a" * 64
DIGEST_B = "sha256:" + "b" * 64
DIGEST_C = "sha256:" + "c" * 64


def base_model_identity() -> BaseModelIdentity:
    return BaseModelIdentity(
        model_name="org/model",
        model_revision=REVISION,
        model_config_digest=DIGEST_A,
        tokenizer_name="org/model",
        tokenizer_revision=REVISION,
        tokenizer_digest=DIGEST_B,
        chat_template_digest=DIGEST_C,
        vocab_size=128,
    )


def adapter_manifest() -> AdapterManifest:
    return create_adapter_manifest(
        files=(
            AdapterFile(name="adapter_config.json", size_bytes=100, digest=DIGEST_A),
            AdapterFile(name="adapter_model.safetensors", size_bytes=1000, digest=DIGEST_B),
        ),
        rank=8,
        alpha=16,
        target_modules=("k_proj", "q_proj"),
    )


def policy_manifest(version: int = 1, created_at: float = 10.0) -> PolicyManifest:
    return PolicyManifest(
        run_id="run-1",
        policy_version=version,
        base_model=base_model_identity(),
        adapter=None if version == 0 else adapter_manifest(),
        created_at=created_at,
    )


def rollout_assignment() -> RolloutAssignment:
    return RolloutAssignment(
        assignment_id="assignment-1",
        group_id="group-1",
        group_index=0,
        group_size=2,
        kind="train",
        environment=EnvironmentIdentity(id="environment-v1", revision="1.0.0"),
        task_data={"idx": 1, "prompt": "solve this"},
        sampling=SamplingConfig(max_tokens=32),
        policy=policy_manifest(),
        created_at=10.0,
        deadline_at=30.0,
    )


def test_protocol_models_forbid_unknown_fields_and_are_frozen():
    with pytest.raises(ValidationError):
        BaseModelIdentity.model_validate({**base_model_identity().model_dump(), "unknown": True})

    identity = base_model_identity()
    with pytest.raises(ValidationError):
        identity.vocab_size = 256


def test_canonical_json_and_digest_are_deterministic():
    left = canonical_json_bytes({"b": 2, "a": {"y": 2, "x": 1}})
    right = canonical_json_bytes({"a": {"x": 1, "y": 2}, "b": 2})
    assert left == right == b'{"a":{"x":1,"y":2},"b":2}'
    assert sha256_digest(left) == sha256_digest(right)


def test_adapter_manifest_validates_content_digest_and_order():
    manifest = adapter_manifest()
    assert manifest.digest.startswith("sha256:")

    with pytest.raises(ValidationError, match="digest"):
        AdapterManifest(**{**manifest.model_dump(), "digest": DIGEST_C})
    with pytest.raises(ValidationError, match="ordered exactly"):
        create_adapter_manifest(
            files=tuple(reversed(manifest.files)),
            rank=manifest.rank,
            alpha=manifest.alpha,
            target_modules=manifest.target_modules,
        )


def test_policy_identity_is_immutable_and_ignores_publication_time():
    first = policy_manifest(created_at=10.0)
    second = policy_manifest(created_at=20.0)
    assert first.policy_id == second.policy_id
    assert first.served_model_name == first.policy_id
    assert policy_manifest_digest(first) == policy_manifest_digest(second)

    base = policy_manifest(version=0)
    assert base.policy_id.startswith("policy-v00000000-")
    assert base.served_model_name == "org/model"

    other_run = PolicyManifest(
        **{**first.model_dump(), "run_id": "run-2", "policy_id": None, "served_model_name": None}
    )
    assert other_run.policy_id != first.policy_id
    assert policy_manifest_digest(other_run) != policy_manifest_digest(first)


def test_policy_version_requires_matching_adapter_state():
    with pytest.raises(ValidationError, match="version 0"):
        PolicyManifest(
            run_id="run-1",
            policy_version=0,
            base_model=base_model_identity(),
            adapter=adapter_manifest(),
            created_at=1.0,
        )
    with pytest.raises(ValidationError, match="include an adapter"):
        PolicyManifest(
            run_id="run-1",
            policy_version=1,
            base_model=base_model_identity(),
            created_at=1.0,
        )


def test_assignment_and_lease_validate_ranges_and_time_ordering():
    assignment = rollout_assignment()
    lease = AssignmentLease(
        lease_id="lease-1",
        attempt=1,
        worker_id="worker-1",
        worker_session_id="session-1",
        issued_at=20.0,
        expires_at=30.0,
        assignment=assignment,
    )
    assert lease.assignment.policy.policy_id == assignment.policy.policy_id

    with pytest.raises(ValidationError, match="group_index"):
        RolloutAssignment(**{**assignment.model_dump(), "group_index": 2})
    with pytest.raises(ValidationError, match="expires_at"):
        AssignmentLease(**{**lease.model_dump(), "expires_at": 20.0})

    with pytest.raises(ValueError, match="Out of range float values"):
        RolloutAssignment(**{**assignment.model_dump(), "task_data": {"score": float("nan")}})


def test_worker_capabilities_require_sorted_unique_environments():
    common = {
        "base_model": base_model_identity(),
        "runtime": {
            "aether_rl_version": "0.7.0",
            "python_version": "3.12.0",
            "torch_version": "2.11.0",
            "transformers_version": "5.6.2",
            "vllm_version": "0.24.0",
        },
        "max_concurrent_assignments": 1,
        "gpu_count": 1,
        "tensor_parallel_size": 1,
    }
    WorkerCapabilities(
        **common,
        environments=(
            EnvironmentIdentity(id="a", revision="1"),
            EnvironmentIdentity(id="b", revision="1"),
        ),
    )
    with pytest.raises(ValidationError, match="sorted"):
        WorkerCapabilities(
            **common,
            environments=(
                EnvironmentIdentity(id="b", revision="1"),
                EnvironmentIdentity(id="a", revision="1"),
            ),
        )


def test_terminal_envelope_discriminator_roundtrips_result_and_failure():
    adapter = TypeAdapter(TerminalEnvelope)
    result = ResultEnvelope(
        assignment_id="assignment-1",
        attempt=1,
        lease_id="lease-1",
        worker_id="worker-1",
        worker_session_id="session-1",
        requested_policy_id=policy_manifest().policy_id,
        served_policy_id=policy_manifest().policy_id,
        requested_policy_digest=policy_manifest_digest(policy_manifest()),
        served_policy_digest=policy_manifest_digest(policy_manifest()),
        completed_at=20.0,
        result_digest=episode_digest(WireEpisode(id="episode-1", env="environment-v1", ok=True)),
        episode=WireEpisode(id="episode-1", env="environment-v1", ok=True),
    )
    decoded_result = adapter.validate_python(result.model_dump())
    assert isinstance(decoded_result, ResultEnvelope)
    assert decoded_result.episode.id == "episode-1"

    with pytest.raises(ValidationError, match="served policy ID"):
        ResultEnvelope(**{**result.model_dump(), "served_policy_id": "policy-other"})
    with pytest.raises(ValidationError, match="episode payload"):
        ResultEnvelope(**{**result.model_dump(), "result_digest": DIGEST_B})

    result.episode.ok = False
    with pytest.raises(PydanticSerializationError, match="result_digest does not match"):
        result.model_dump_json()

    failure = FailureEnvelope(
        assignment_id="assignment-1",
        attempt=1,
        lease_id="lease-1",
        worker_id="worker-1",
        worker_session_id="session-1",
        failed_at=20.0,
        code="environment-error",
        message="environment failed",
        retryable=True,
    )
    assert isinstance(adapter.validate_python(failure.model_dump()), FailureEnvelope)
