# Distributed up repetition

This run trains Qwen2.5-0.5B-Instruct for 50 GRPO steps using the same prompt for every rollout: `Write a super short story about a boy's morning routine.` A response earns `0.02` for each literal lowercase `up` substring, capped at a reward of `1.0` for 50 or more occurrences.

The trainer retains a complete checkpoint every step. Start with at least 130 GiB free on the filesystem containing the run root.

Install and launch the server:

```bash
scripts/setup-server.sh up-story-v1
scripts/run-server.sh examples/distributed/up-50step/server.toml
```

Install and launch each worker with the same coordinator token:

```bash
scripts/setup-worker.sh up-story-v1
scripts/run-worker.sh \
  examples/distributed/up-50step/worker.toml \
  https://coordinator.example.com
```

Publish eval metrics to W&B from the server:

```bash
uv run --no-default-groups eval-report \
  --run-root outputs/up-qwen25-50step/server \
  --source-id up-50step-eval \
  --wandb-project aether-rl-up \
  --wandb-name qwen2.5-0.5b-up-50step-eval \
  --wandb-group distributed-learning-proof \
  --watch-seconds 20
```
