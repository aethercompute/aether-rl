import msgspec

from aether_rl.coordinator.results import ProcessedGroupPayload
from aether_rl.entrypoints.eval_report import summarize_eval


def test_summarize_eval_groups_by_behavior_policy(tmp_path):
    groups_dir = tmp_path / "groups"
    groups_dir.mkdir()
    payload = ProcessedGroupPayload(
        group_id="group-1",
        source_id="reverse-text-proof-eval",
        kind="eval",
        policy_id="policy-3",
        policy_version=3,
        input_digest="sha256:" + "1" * 64,
        rollouts=[],
        evaluation_records=[
            {
                "ok": True,
                "rewards": {"character_accuracy": 1.0},
                "metrics": {"exact_match": 1.0, "exact_format": 1.0, "length_accuracy": 1.0},
            },
            {
                "ok": True,
                "rewards": {"character_accuracy": 0.0},
                "metrics": {"exact_match": 0.0, "exact_format": 0.0, "length_accuracy": 1.0},
            },
            {"ok": False, "rewards": {"character_accuracy": 1.0}, "metrics": {}},
        ],
    )
    (groups_dir / "group.msgpack").write_bytes(msgspec.msgpack.encode(payload))

    assert summarize_eval(groups_dir, source_id="reverse-text-proof-eval") == [
        {
            "source_id": "reverse-text-proof-eval",
            "policy_version": 3,
            "policy_id": "policy-3",
            "groups": 1,
            "rollouts": 3,
            "errors": 1,
            "mean_reward": 1 / 3,
            "effective_rollouts": 2,
            "effective_mean_reward": 0.5,
            "exact_match_mean": 0.5,
            "exact_format_mean": 0.5,
            "length_accuracy_mean": 1.0,
        }
    ]
