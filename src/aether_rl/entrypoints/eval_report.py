import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import msgspec

from aether_rl.coordinator.results import ProcessedGroupPayload

EVAL_METRICS = ("exact_match", "exact_format", "length_accuracy")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize durable distributed evaluation results by policy")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-id")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name", default="distributed-eval")
    parser.add_argument("--wandb-group")
    parser.add_argument("--watch-seconds", type=float)
    args = parser.parse_args()
    if args.watch_seconds is not None and args.watch_seconds <= 0:
        parser.error("--watch-seconds must be positive")

    groups_dir = args.run_root / "training-queue" / "groups"
    if not groups_dir.is_dir():
        parser.error(f"processed group directory does not exist: {groups_dir}")

    run = None
    if args.wandb_project is not None:
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            group=args.wandb_group,
            tags=["distributed", "evaluation"],
        )
        run.define_metric("eval/policy_version")
        run.define_metric("eval/*", step_metric="eval/policy_version")

    last_logged: dict[tuple[str, int, str], dict[str, Any]] = {}
    try:
        while True:
            policies = summarize_eval(groups_dir, source_id=args.source_id)
            print(json.dumps({"run_root": str(args.run_root), "policies": policies}, indent=2), flush=True)
            if run is not None:
                for policy in policies:
                    key = (policy["source_id"], policy["policy_version"], policy["policy_id"])
                    if last_logged.get(key) == policy:
                        continue
                    run.log(
                        {
                            "eval/policy_version": policy["policy_version"],
                            "eval/reward": policy["mean_reward"],
                            "eval/exact_match": policy["exact_match_mean"],
                            "eval/exact_format": policy["exact_format_mean"],
                            "eval/length_accuracy": policy["length_accuracy_mean"],
                            "eval/rollouts": policy["rollouts"],
                            "eval/errors": policy["errors"],
                        }
                    )
                    last_logged[key] = policy.copy()
            if args.watch_seconds is None:
                break
            time.sleep(args.watch_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        if run is not None:
            run.finish()


def summarize_eval(groups_dir: Path, *, source_id: str | None = None) -> list[dict[str, Any]]:

    totals: dict[tuple[str, int, str], dict[str, Any]] = defaultdict(
        lambda: {
            "groups": 0,
            "rollouts": 0,
            "errors": 0,
            "reward_sum": 0.0,
            "effective_rollouts": 0,
            "effective_reward_sum": 0.0,
            "metric_counts": defaultdict(int),
            "metric_sums": defaultdict(float),
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
            metrics = record.get("metrics", {})
            for metric in EVAL_METRICS:
                if metric in metrics:
                    aggregate["metric_counts"][metric] += 1
                    aggregate["metric_sums"][metric] += float(metrics[metric])

    policies: list[dict[str, Any]] = []
    for (source_id, policy_version, policy_id), aggregate in sorted(totals.items()):
        rollouts = aggregate["rollouts"]
        effective = aggregate["effective_rollouts"]
        policy = {
            "source_id": source_id,
            "policy_version": policy_version,
            "policy_id": policy_id,
            "groups": aggregate["groups"],
            "rollouts": rollouts,
            "errors": aggregate["errors"],
            "mean_reward": aggregate["reward_sum"] / rollouts if rollouts else None,
            "effective_rollouts": effective,
            "effective_mean_reward": aggregate["effective_reward_sum"] / effective if effective else None,
        }
        for metric in EVAL_METRICS:
            count = aggregate["metric_counts"][metric]
            policy[f"{metric}_mean"] = aggregate["metric_sums"][metric] / count if count else None
        policies.append(policy)

    return policies


if __name__ == "__main__":
    main()
