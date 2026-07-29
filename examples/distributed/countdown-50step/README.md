# Distributed countdown arithmetic

This run trains Qwen2.5-0.5B-Instruct for 50 GRPO steps on guaranteed-solvable countdown arithmetic tasks. The train and eval streams are deterministic, infinite, and disjoint. A response earns `0.1` for using every provided number exactly once and the full `1.0` only when the resulting expression reaches the target.

The trainer retains a complete checkpoint every step. Start with at least 130 GiB free on the filesystem containing the run root.

Install and launch the server:

```bash
scripts/setup-server.sh countdown-proof-v1
scripts/run-server.sh examples/distributed/countdown-50step/server.toml
```

Install and launch each worker with the same coordinator token:

```bash
scripts/setup-worker.sh countdown-proof-v1
scripts/run-worker.sh \
  examples/distributed/countdown-50step/worker.toml \
  https://coordinator.example.com
```

Publish eval metrics to W&B from the server:

```bash
uv run --no-default-groups eval-report \
  --run-root outputs/countdown-qwen25-50step/server \
  --source-id countdown-50step-eval \
  --wandb-project aether-rl-countdown \
  --wandb-name qwen2.5-0.5b-countdown-50step-eval \
  --wandb-group distributed-learning-proof \
  --watch-seconds 20
```
