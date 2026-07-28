# Distributed reverse-text proof

This experiment validates learning and distributed execution with one trainer and one or more rollout workers. It trains Qwen2.5-0.5B-Instruct to reverse deterministic random lowercase strings of length 3–5.

The training and evaluation splits are generated locally from different seeds and do not overlap. Training uses dense character-position reward; evaluation reports mean reward, exact match, exact format, and correct output length for every policy version.

The default run performs 10 LoRA optimizer steps. Each update consumes 64 surviving train rollouts, grouped into eight-sample GRPO groups. Completions are capped at 12 tokens, checkpoints are retained every step, and optional full-weight exports are disabled.

## Setup

On the trainer/coordinator machine:

```bash
scripts/setup-server.sh reverse-text-proof-v1
export AETHER_COORDINATOR_TOKEN='<random-ascii-secret>'
export WANDB_API_KEY='<your-key>'
```

On every rollout worker:

```bash
scripts/setup-worker.sh reverse-text-proof-v1
export AETHER_COORDINATOR_TOKEN='<same-random-ascii-secret>'
```

For remote workers, expose the coordinator through authenticated HTTPS and replace `coordinator_url` in a local worker config. Never expose the coordinator's plain HTTP port directly to the internet.

## Launch

Start the trainer and coordinator:

```bash
scripts/run-server.sh examples/distributed/reverse-text-proof/server.toml
```

Start the same worker command on each rollout machine, passing the coordinator's authenticated HTTPS URL:

```bash
scripts/run-worker.sh examples/distributed/reverse-text-proof/worker.toml https://coordinator.example.com
```

Follow evaluation progress in a separate terminal:

```bash
uv run --no-default-groups eval-report \
  --run-root outputs/reverse-text-qwen25-proof/server \
  --source-id reverse-text-proof-eval \
  --watch-seconds 10
```

Add `--wandb-project aether-rl-reverse-text` to publish the evaluation series as a separate W&B run. Compare `eval/exact_match` and `eval/reward` against `eval/policy_version`; `eval/exact_format` and `eval/length_accuracy` distinguish formatting failures from reversal errors.

Policy 0 is the unmodified base model. Use policy 9 for the predefined comparison because terminal policy 10 normally receives no new eval leases. The learning proof passes when both policies have at least 12 eval rollouts, policy 9 improves `mean_reward` by at least 0.15, policy 9 improves `exact_match_mean` by at least 0.15, format and length accuracy remain at least 0.95, and eval errors remain zero or are operationally explained. Policies, checkpoints, and trainer metrics must also advance with optimizer steps.
