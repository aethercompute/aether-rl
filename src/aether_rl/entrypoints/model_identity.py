import argparse
import re
from pathlib import Path

import tomli_w

from aether_rl.worker.identity import calculate_base_model_identity


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pinned Aether RL base-model identity")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-name")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--hf-cache-dir", type=Path)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    tokenizer_name = args.tokenizer_name or args.model_name
    tokenizer_revision = args.tokenizer_revision or args.model_revision
    for name, revision in (("model", args.model_revision), ("tokenizer", tokenizer_revision)):
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            parser.error(f"{name} revision must be a full 40-character lowercase commit hash")

    cache_dir = None if args.hf_cache_dir is None else str(args.hf_cache_dir / "hub")
    identity = calculate_base_model_identity(
        model_name=args.model_name,
        model_revision=args.model_revision,
        tokenizer_name=tokenizer_name,
        tokenizer_revision=tokenizer_revision,
        cache_dir=cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    print(tomli_w.dumps({"base_model": identity.model_dump(mode="python")}), end="")


if __name__ == "__main__":
    main()
