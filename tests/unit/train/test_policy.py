from pathlib import Path

import pytest
import torch

import aether_rl.trainer.policy as policy_module
from aether_rl.protocol import BaseModelIdentity
from aether_rl.trainer.policy import publish_lora_policy, verify_lora_policy
from aether_rl.trainer.runs import MultiRunManager

REVISION = "a" * 40
DIGEST = "sha256:" + "b" * 64


def base_model_identity() -> BaseModelIdentity:
    return BaseModelIdentity(
        model_name="org/model",
        model_revision=REVISION,
        model_config_digest=DIGEST,
        tokenizer_name="org/model",
        tokenizer_revision=REVISION,
        tokenizer_digest=DIGEST,
        chat_template_digest=DIGEST,
        vocab_size=128,
    )


def adapter_state_dict() -> dict[str, torch.Tensor]:
    return {
        "model.layers.0.self_attn.q_proj.lora_A.weight": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "model.layers.0.self_attn.q_proj.lora_B.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
    }


def publish(path: Path, *, created_at: float = 10.0):
    return publish_lora_policy(
        path,
        run_id="run-1",
        policy_version=1,
        base_model=base_model_identity(),
        state_dict=adapter_state_dict(),
        rank=2,
        alpha=4,
        dropout=0.0,
        created_at=created_at,
    )


def test_publish_lora_policy_is_atomic_verifiable_and_idempotent(tmp_path: Path):
    manifest = publish(tmp_path)
    policy_path = tmp_path / manifest.policy_id

    assert {entry.name for entry in policy_path.iterdir()} == {
        "adapter_config.json",
        "adapter_model.safetensors",
        "manifest.json",
    }
    assert verify_lora_policy(policy_path) == manifest
    assert publish(tmp_path) == manifest
    assert publish(tmp_path, created_at=11.0) == manifest
    assert not any(entry.name.startswith(".policy-") for entry in tmp_path.iterdir())


@pytest.mark.parametrize("filename", ["adapter_config.json", "adapter_model.safetensors"])
def test_verify_lora_policy_rejects_corruption(tmp_path: Path, filename: str):
    manifest = publish(tmp_path)
    artifact = tmp_path / manifest.policy_id / filename
    contents = artifact.read_bytes()
    artifact.write_bytes(bytes([contents[0] ^ 1]) + contents[1:])

    with pytest.raises(ValueError, match="digest does not match"):
        verify_lora_policy(tmp_path / manifest.policy_id)


def test_publish_lora_policy_rejects_non_lora_weights_without_artifacts(tmp_path: Path):
    with pytest.raises(ValueError, match="unsupported non-LoRA"):
        publish_lora_policy(
            tmp_path,
            run_id="run-1",
            policy_version=1,
            base_model=base_model_identity(),
            state_dict={"model.embed_tokens.weight": torch.ones(2, 2)},
            rank=2,
            alpha=4,
            dropout=0.0,
            created_at=10.0,
        )

    assert not any(tmp_path.iterdir())


def test_publish_lora_policy_rejects_invalid_dropout(tmp_path: Path):
    with pytest.raises(ValueError, match="dropout must be between"):
        publish_lora_policy(
            tmp_path,
            run_id="run-1",
            policy_version=1,
            base_model=base_model_identity(),
            state_dict=adapter_state_dict(),
            rank=2,
            alpha=4,
            dropout=-0.1,
            created_at=10.0,
        )


def test_publish_lora_policy_cleans_temporary_directory_on_write_failure(tmp_path: Path, monkeypatch):
    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(policy_module, "save_file", fail_save)

    with pytest.raises(OSError, match="disk full"):
        publish(tmp_path)

    assert not any(tmp_path.iterdir())


def test_publish_lora_policy_rejects_incomplete_or_wrong_rank_tensors(tmp_path: Path):
    state_dict = adapter_state_dict()
    state_dict.pop("model.layers.0.self_attn.q_proj.lora_B.weight")
    with pytest.raises(ValueError, match="pair is incomplete"):
        publish_lora_policy(
            tmp_path,
            run_id="run-1",
            policy_version=1,
            base_model=base_model_identity(),
            state_dict=state_dict,
            rank=2,
            alpha=4,
            dropout=0.0,
            created_at=10.0,
        )

    state_dict = adapter_state_dict()
    state_dict["model.layers.0.self_attn.q_proj.lora_A.weight"] = torch.ones(3, 4)
    state_dict["model.layers.0.self_attn.q_proj.lora_B.weight"] = torch.ones(4, 3)
    with pytest.raises(ValueError, match="rank does not match manifest"):
        publish_lora_policy(
            tmp_path,
            run_id="run-1",
            policy_version=1,
            base_model=base_model_identity(),
            state_dict=state_dict,
            rank=2,
            alpha=4,
            dropout=0.0,
            created_at=10.0,
        )


@pytest.mark.parametrize(
    "state_dict",
    [
        {
            "model.layers.0.mlp.experts.0.up_proj.lora_A.weight": torch.ones(2, 4),
            "model.layers.0.mlp.experts.0.up_proj.lora_B.weight": torch.ones(8, 2),
            "model.layers.0.mlp.experts.0.down_proj.lora_A.weight": torch.ones(2, 8),
            "model.layers.0.mlp.experts.0.down_proj.lora_B.weight": torch.ones(4, 2),
        },
        {
            "model.layers.0.mlp.experts.base_layer.lora_A.weight": torch.ones(6, 4),
            "model.layers.0.mlp.experts.base_layer.lora_B.weight": torch.ones(16, 6),
            "model.layers.0.mlp.experts.lora_A.weight": torch.ones(6, 8),
            "model.layers.0.mlp.experts.lora_B.weight": torch.ones(4, 6),
        },
    ],
)
def test_publish_lora_policy_accepts_supported_moe_layouts(tmp_path: Path, state_dict: dict[str, torch.Tensor]):
    manifest = publish_lora_policy(
        tmp_path,
        run_id="run-1",
        policy_version=1,
        base_model=base_model_identity(),
        state_dict=state_dict,
        rank=2,
        alpha=4,
        dropout=0.0,
        created_at=10.0,
    )

    assert verify_lora_policy(tmp_path / manifest.policy_id) == manifest


def test_publish_lora_policy_rejects_inconsistent_stacked_moe_layout(tmp_path: Path):
    state_dict = {
        "model.layers.0.mlp.experts.base_layer.lora_A.weight": torch.ones(6, 4),
        "model.layers.0.mlp.experts.base_layer.lora_B.weight": torch.ones(8, 6),
        "model.layers.0.mlp.experts.lora_A.weight": torch.ones(6, 8),
        "model.layers.0.mlp.experts.lora_B.weight": torch.ones(4, 6),
    }

    with pytest.raises(ValueError, match="intermediate sizes do not match"):
        publish_lora_policy(
            tmp_path,
            run_id="run-1",
            policy_version=1,
            base_model=base_model_identity(),
            state_dict=state_dict,
            rank=2,
            alpha=4,
            dropout=0.0,
            created_at=10.0,
        )


def test_publish_lora_policy_rejects_incomplete_per_expert_moe_layout(tmp_path: Path):
    state_dict = {
        "model.layers.0.mlp.experts.0.gate_proj.lora_A.weight": torch.ones(2, 4),
        "model.layers.0.mlp.experts.0.gate_proj.lora_B.weight": torch.ones(8, 2),
    }

    with pytest.raises(ValueError, match="projections are incomplete"):
        publish_lora_policy(
            tmp_path,
            run_id="run-1",
            policy_version=1,
            base_model=base_model_identity(),
            state_dict=state_dict,
            rank=2,
            alpha=4,
            dropout=0.0,
            created_at=10.0,
        )


def test_publish_lora_policy_rejects_unknown_moe_layout(tmp_path: Path):
    state_dict = {
        "model.layers.0.mlp.experts.foo.up_proj.lora_A.weight": torch.ones(2, 4),
        "model.layers.0.mlp.experts.foo.up_proj.lora_B.weight": torch.ones(8, 2),
    }

    with pytest.raises(ValueError, match="unsupported MoE"):
        publish_lora_policy(
            tmp_path,
            run_id="run-1",
            policy_version=1,
            base_model=base_model_identity(),
            state_dict=state_dict,
            rank=2,
            alpha=4,
            dropout=0.0,
            created_at=10.0,
        )


def test_adapter_export_must_include_every_optimized_parameter():
    parameter = torch.nn.Parameter(torch.ones(2))

    class IncompleteAdapter:
        def named_parameters_for_adapter(self, idx: int):
            return [("lora_A", parameter)]

        def state_dict_for_adapter(self, idx: int):
            return {"lora_A.weight": parameter[:1]}

    manager = MultiRunManager.__new__(MultiRunManager)
    manager._modules = [("layer", IncompleteAdapter())]
    manager._adapter_state_dict_converter = None

    with pytest.raises(ValueError, match="do not match optimized parameters"):
        manager.get_state_dict_for_run(0)
