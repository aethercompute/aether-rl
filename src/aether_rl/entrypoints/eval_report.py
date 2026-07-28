import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import msgspec

from aether_rl.coordinator.results import ProcessedGroupPayload


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize durable distributed evaluation results by policy")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-id")
    args = parser.parse_args()

    groups_dir = args.run_root / "training-queue" / "groups"
    if not groups_dir.is_dir():
        parser.error(f"processed group directory does not exist: {groups_dir}")

    policies = summarize_eval(groups_dir, source_id=args.source_id)
    print(json.dumps({"run_root": str(args.run_root), "policies": policies}, indent=2))


def summarize_eval(groups_dir: Path, *, source_id: str | None = None) -> list[dict[str, Any]]:

    totals: dict[tuple[str, int, str], dict[str, Any]] = defaultdict(
        lambda: {
            "groups": 0,
            "rollouts": 0,
            "errors": 0,
            "reward_sum": 0.0,
            "effective_rollouts": 0,
            "effective_reward_sum": 0.0,
            "exact_format_count": 0,
            "exact_format_sum": 0.0,
        }
    )
    for path in sorted(groups_dir.glob("*.msgpack")):
        payload = msgspec.msgpack.decode(path.read_bytes(), type=ProcessedGroupPayload)
        if payload.kind != "eval" or (source_id is not None and payload.source_id != source_id):
            continue
        key = (payload.source_id, payload.policy_version, payload.policy_id)
        aggregate = totals[key]
        aggregate["groups"] += 1
        for record in payload.evaluation_records:
            reward = sum(float(value) for value in record.get("rewards", {}).values())
            ok = bool(record.get("ok", False))
            aggregate["rollouts"] += 1
            aggregate["reward_sum"] += reward if ok else 0.0
            aggregate["errors"] += int(not ok)
            if ok:
                aggregate["effective_rollouts"] += 1
                aggregate["effective_reward_sum"] += reward
            exact_format = record.get("metrics", {}).get("exact_format")
            if exact_format is not None:
                aggregate["exact_format_count"] += 1
                aggregate["exact_format_sum"] += float(exact_format)

    policies: list[dict[str, Any]] = []
    for (source_id, policy_version, policy_id), aggregate in sorted(totals.items()):
        rollouts = aggregate["rollouts"]
        effective = aggregate["effective_rollouts"]
        exact_count = aggregate["exact_format_count"]
        policies.append(
            {
                "source_id": source_id,
                "policy_version": policy_version,
                "policy_id": policy_id,
                "groups": aggregate["groups"],
                "rollouts": rollouts,
                "errors": aggregate["errors"],
                "mean_reward": aggregate["reward_sum"] / rollouts if rollouts else None,
                "effective_rollouts": effective,
                "effective_mean_reward": aggregate["effective_reward_sum"] / effective if effective else None,
                "exact_format_mean": aggregate["exact_format_sum"] / exact_count if exact_count else None,
            }
        )

    return policies


if __name__ == "__main__":
    main()
