from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from aether_rl.configs.worker import WorkerConfig
from aether_rl.protocol import BaseModelIdentity, canonical_json_bytes, sha256_digest


def discover_base_model_identity(config: WorkerConfig) -> BaseModelIdentity:
    expected = BaseModelIdentity.model_validate(config.base_model.model_dump(mode="python"))
    if expected.quantization != "none":
        raise ValueError("worker serving currently supports only quantization='none'")
    cache_dir = None if config.hf_cache_dir is None else str(config.hf_cache_dir / "hub")
    config_path = Path(
        hf_hub_download(
            repo_id=expected.model_name,
            filename="config.json",
            revision=expected.model_revision,
            cache_dir=cache_dir,
        )
    )
    model_config = json.loads(config_path.read_bytes())
    if model_config.get("quantization_config") is not None:
        raise ValueError("model repository is quantized but worker identity requires unquantized weights")
    tokenizer = AutoTokenizer.from_pretrained(
        expected.tokenizer_name,
        revision=expected.tokenizer_revision,
        cache_dir=cache_dir,
        trust_remote_code=config.trust_remote_code,
    )
    if tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    discovered = BaseModelIdentity(
        model_name=expected.model_name,
        model_revision=expected.model_revision,
        model_config_digest=sha256_digest(canonical_json_bytes(model_config)),
        tokenizer_name=expected.tokenizer_name,
        tokenizer_revision=expected.tokenizer_revision,
        tokenizer_digest=tokenizer_digest(tokenizer),
        chat_template_digest=chat_template_digest(tokenizer.chat_template),
        vocab_size=len(tokenizer),
        quantization="none",
    )
    if discovered != expected:
        mismatches = [
            field for field in BaseModelIdentity.model_fields if getattr(discovered, field) != getattr(expected, field)
        ]
        raise ValueError(f"configured base model identity does not match pinned artifacts: {', '.join(mismatches)}")
    return discovered


def tokenizer_digest(tokenizer: Any) -> str:
    added_tokens = []
    for token_id, token in sorted(tokenizer.added_tokens_decoder.items()):
        added_tokens.append(
            {
                "id": token_id,
                "content": token.content,
                "single_word": token.single_word,
                "lstrip": token.lstrip,
                "rstrip": token.rstrip,
                "normalized": token.normalized,
                "special": token.special,
            }
        )
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_payload = None if backend is None else json.loads(backend.to_str())
    payload = {
        "schema_version": 1,
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "backend": backend_payload,
        "vocab": sorted(tokenizer.get_vocab().items()),
        "added_tokens": added_tokens,
        "special_tokens_map": _json_value(tokenizer.special_tokens_map),
        "special_token_ids": {
            name: getattr(tokenizer, f"{name}_id", None)
            for name in ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token")
        },
        "base_vocab_size": tokenizer.vocab_size,
        "total_vocab_size": len(tokenizer),
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "model_input_names": list(tokenizer.model_input_names),
    }
    return sha256_digest(canonical_json_bytes(payload))


def chat_template_digest(template: Any) -> str:
    return sha256_digest(canonical_json_bytes({"schema_version": 1, "chat_template": _json_value(template)}))


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "content"):
        return value.content
    return str(value)
