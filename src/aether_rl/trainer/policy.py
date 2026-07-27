from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from pathlib import Path

from pydantic import ValidationError
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from aether_rl.protocol import (
    AdapterFile,
    BaseModelIdentity,
    PolicyManifest,
    canonical_json_bytes,
    create_adapter_manifest,
    policy_manifest_digest,
)

ADAPTER_CONFIG_NAME = "adapter_config.json"
ADAPTER_WEIGHTS_NAME = "adapter_model.safetensors"
POLICY_MANIFEST_NAME = "manifest.json"
POLICY_FILE_NAMES = frozenset({ADAPTER_CONFIG_NAME, ADAPTER_WEIGHTS_NAME, POLICY_MANIFEST_NAME})


def publish_lora_policy(
    policies_dir: Path,
    *,
    run_id: str,
    policy_version: int,
    base_model: BaseModelIdentity,
    state_dict: dict[str, Tensor],
    rank: int,
    alpha: float,
    dropout: float,
    created_at: float,
) -> PolicyManifest:
    """Atomically publish one immutable, PEFT-compatible LoRA policy."""
    if policy_version < 1:
        raise ValueError("published LoRA policies must have a positive policy version")
    if not math.isfinite(dropout) or not 0 <= dropout <= 1:
        raise ValueError("LoRA dropout must be between 0 and 1")
    target_modules = _validate_adapter_tensors(state_dict, rank)
    adapter_config = {
        "alpha_pattern": {},
        "base_model_name_or_path": base_model.model_name,
        "bias": "none",
        "inference_mode": True,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "rank_pattern": {},
        "revision": base_model.model_revision,
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM",
    }

    try:
        parent_mode = policies_dir.parent.stat().st_mode
    except FileNotFoundError as error:
        raise ValueError("policy artifact parent directory must already exist") from error
    if not stat.S_ISDIR(parent_mode):
        raise ValueError("policy artifact parent must be a directory")
    policies_dir_existed = policies_dir.exists()
    policies_dir.mkdir(exist_ok=True)
    if not policies_dir_existed:
        _fsync_directory(policies_dir.parent)
    temporary_path = Path(tempfile.mkdtemp(prefix=".policy-", dir=policies_dir))
    try:
        weights_path = temporary_path / ADAPTER_WEIGHTS_NAME
        save_file(
            {name: tensor.detach().cpu().contiguous() for name, tensor in state_dict.items()},
            weights_path,
            metadata={"format": "pt"},
        )
        _fsync_file(weights_path)

        config_path = temporary_path / ADAPTER_CONFIG_NAME
        _write_durable(config_path, canonical_json_bytes(adapter_config))

        adapter = create_adapter_manifest(
            files=(
                _adapter_file(config_path),
                _adapter_file(weights_path),
            ),
            rank=rank,
            alpha=alpha,
            target_modules=target_modules,
        )
        policy = PolicyManifest(
            run_id=run_id,
            policy_version=policy_version,
            base_model=base_model,
            adapter=adapter,
            created_at=created_at,
        )
        _write_durable(temporary_path / POLICY_MANIFEST_NAME, canonical_json_bytes(policy))
        verify_lora_policy(temporary_path, expected=policy)
        _fsync_directory(temporary_path)

        final_path = policies_dir / policy.policy_id
        if final_path.exists():
            existing = verify_lora_policy(final_path, expected=policy)
            _fsync_directory(policies_dir)
            return existing
        try:
            temporary_path.rename(final_path)
        except OSError as error:
            if error.errno not in {errno.EEXIST, errno.ENOTEMPTY}:
                raise
            existing = verify_lora_policy(final_path, expected=policy)
            _fsync_directory(policies_dir)
            return existing
        _fsync_directory(policies_dir)
        return policy
    finally:
        if temporary_path.exists():
            shutil.rmtree(temporary_path)


def verify_lora_policy(path: Path, *, expected: PolicyManifest | None = None) -> PolicyManifest:
    """Verify an immutable policy directory before serving or loading it."""
    if not path.is_dir():
        raise ValueError(f"policy artifact is not a directory: {path}")
    entries = {entry.name for entry in path.iterdir()}
    if entries != POLICY_FILE_NAMES:
        raise ValueError(f"policy artifact must contain exactly {sorted(POLICY_FILE_NAMES)}")
    for name in POLICY_FILE_NAMES:
        artifact_path = path / name
        if artifact_path.is_symlink() or not artifact_path.is_file():
            raise ValueError(f"policy artifact must contain regular files: {name}")

    manifest_bytes = (path / POLICY_MANIFEST_NAME).read_bytes()
    try:
        policy = PolicyManifest.model_validate_json(manifest_bytes)
    except ValidationError as error:
        raise ValueError("invalid policy manifest") from error
    if manifest_bytes != canonical_json_bytes(policy):
        raise ValueError("policy manifest is not canonical JSON")
    if policy.adapter is None:
        raise ValueError("published policy must contain an adapter")
    if path.name != policy.policy_id and not path.name.startswith(".policy-"):
        raise ValueError("policy directory name does not match policy ID")
    if expected is not None and policy_manifest_digest(policy) != policy_manifest_digest(expected):
        raise ValueError("existing policy does not match the requested policy")

    for file in policy.adapter.files:
        file_path = path / file.name
        if file_path.stat().st_size != file.size_bytes:
            raise ValueError(f"adapter file size does not match manifest: {file.name}")
        if _file_digest(file_path) != file.digest:
            raise ValueError(f"adapter file digest does not match manifest: {file.name}")

    config = json.loads((path / ADAPTER_CONFIG_NAME).read_bytes())
    expected_config_keys = {
        "alpha_pattern",
        "base_model_name_or_path",
        "bias",
        "inference_mode",
        "lora_alpha",
        "lora_dropout",
        "modules_to_save",
        "peft_type",
        "r",
        "rank_pattern",
        "revision",
        "target_modules",
        "task_type",
    }
    if not isinstance(config, dict) or set(config) != expected_config_keys:
        raise ValueError("adapter config has unsupported fields")
    if (
        config["peft_type"] != "LORA"
        or config["task_type"] != "CAUSAL_LM"
        or config["bias"] != "none"
        or config["inference_mode"] is not True
        or config["alpha_pattern"] != {}
        or config["rank_pattern"] != {}
    ):
        raise ValueError("adapter config has unsupported LoRA settings")
    dropout = config["lora_dropout"]
    if not isinstance(dropout, int | float) or not math.isfinite(dropout) or not 0 <= dropout <= 1:
        raise ValueError("adapter config has invalid LoRA dropout")
    if config.get("modules_to_save") is not None:
        raise ValueError("modules_to_save is not supported for published policies")
    if config.get("base_model_name_or_path") != policy.base_model.model_name:
        raise ValueError("adapter base model does not match policy manifest")
    if config.get("revision") != policy.base_model.model_revision:
        raise ValueError("adapter base model revision does not match policy manifest")
    if config.get("r") != policy.adapter.rank or config.get("lora_alpha") != policy.adapter.alpha:
        raise ValueError("adapter rank or alpha does not match policy manifest")
    if tuple(config.get("target_modules", ())) != policy.adapter.target_modules:
        raise ValueError("adapter target modules do not match policy manifest")

    with safe_open(path / ADAPTER_WEIGHTS_NAME, framework="pt", device="cpu") as weights:
        tensors = {key: weights.get_tensor(key) for key in weights.keys()}
    if _validate_adapter_tensors(tensors, policy.adapter.rank) != policy.adapter.target_modules:
        raise ValueError("adapter tensor modules do not match policy manifest")
    return policy


def _validate_adapter_tensors(state_dict: dict[str, Tensor], rank: int) -> tuple[str, ...]:
    if not state_dict:
        raise ValueError("LoRA state dict must not be empty")
    pairs: dict[str, dict[str, Tensor]] = {}
    target_modules = set()
    for key, tensor in state_dict.items():
        prefix, marker, suffix = key.rpartition(".lora_")
        if not marker or not prefix or suffix not in {"A.weight", "B.weight"}:
            raise ValueError(f"unsupported non-LoRA adapter tensor: {key}")
        if not tensor.is_floating_point():
            raise ValueError(f"LoRA adapter tensor must use a floating-point dtype: {key}")
        pairs.setdefault(prefix, {})[suffix[0]] = tensor
        target_modules.add(prefix.rsplit(".", 1)[-1])

    for prefix, pair in pairs.items():
        if set(pair) != {"A", "B"}:
            raise ValueError(f"LoRA adapter tensor pair is incomplete: {prefix}")
        lora_a, lora_b = pair["A"], pair["B"]
        if lora_a.ndim != 2 or lora_b.ndim != 2:
            raise ValueError(f"LoRA adapter tensors must be matrices: {prefix}")
        effective_rank = lora_a.shape[0]
        if effective_rank != lora_b.shape[1]:
            raise ValueError(f"LoRA adapter tensor ranks do not match: {prefix}")
        is_stacked_moe = prefix.endswith(".experts") or ".experts.base_layer" in prefix
        if effective_rank != rank and (not is_stacked_moe or effective_rank % rank != 0):
            raise ValueError(f"LoRA adapter tensor rank does not match manifest: {prefix}")

    stacked_roots = {
        prefix.removesuffix(".base_layer")
        for prefix in pairs
        if prefix.endswith(".experts") or prefix.endswith(".experts.base_layer")
    }
    for root in stacked_roots:
        gate_up = pairs.get(f"{root}.base_layer")
        down = pairs.get(root)
        if gate_up is None or down is None:
            raise ValueError(f"stacked MoE LoRA adapter is incomplete: {root}")
        if gate_up["A"].shape[0] != down["A"].shape[0]:
            raise ValueError(f"stacked MoE LoRA adapter expert counts do not match: {root}")
        if gate_up["A"].shape[1] != down["B"].shape[0]:
            raise ValueError(f"stacked MoE LoRA adapter hidden sizes do not match: {root}")
        if gate_up["B"].shape[0] != 2 * down["A"].shape[1]:
            raise ValueError(f"stacked MoE LoRA adapter intermediate sizes do not match: {root}")

    per_expert: dict[str, dict[int, dict[str, dict[str, Tensor]]]] = {}
    for prefix, pair in pairs.items():
        if ".experts." not in prefix:
            continue
        if prefix.endswith(".experts.base_layer"):
            continue
        root, expert_projection = prefix.rsplit(".experts.", 1)
        expert_id, separator, projection = expert_projection.partition(".")
        if not separator or not expert_id.isdigit() or projection not in {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"unsupported MoE LoRA adapter tensor layout: {prefix}")
        per_expert.setdefault(root, {}).setdefault(int(expert_id), {})[projection] = pair

    for root, experts in per_expert.items():
        if set(experts) != set(range(len(experts))):
            raise ValueError(f"per-expert MoE LoRA adapter expert IDs are not contiguous: {root}")
        projection_sets = {frozenset(projections) for projections in experts.values()}
        if len(projection_sets) != 1 or projection_sets.pop() not in {
            frozenset({"up_proj", "down_proj"}),
            frozenset({"gate_proj", "up_proj", "down_proj"}),
        }:
            raise ValueError(f"per-expert MoE LoRA adapter projections are incomplete: {root}")
        reference_shapes = None
        for projections in experts.values():
            up = projections["up_proj"]
            down = projections["down_proj"]
            shapes = {
                f"{projection}.{name}": tuple(tensor.shape)
                for projection, pair in projections.items()
                for name, tensor in pair.items()
            }
            if reference_shapes is None:
                reference_shapes = shapes
            elif shapes != reference_shapes:
                raise ValueError(f"per-expert MoE LoRA adapter shapes do not match: {root}")
            if up["A"].shape[1] != down["B"].shape[0] or up["B"].shape[0] != down["A"].shape[1]:
                raise ValueError(f"per-expert MoE LoRA adapter dimensions do not match: {root}")
            if "gate_proj" in projections:
                gate = projections["gate_proj"]
                if gate["A"].shape != up["A"].shape or gate["B"].shape != up["B"].shape:
                    raise ValueError(f"per-expert MoE gate and up projection shapes do not match: {root}")
    return tuple(sorted(target_modules))


def _adapter_file(path: Path) -> AdapterFile:
    return AdapterFile(name=path.name, size_bytes=path.stat().st_size, digest=_file_digest(path))


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _write_durable(path: Path, data: bytes) -> None:
    with path.open("xb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as file:
        os.fsync(file.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
