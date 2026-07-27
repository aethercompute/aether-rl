# Reverse Text

This example uses SFT followed by RL to train `Qwen3-0.6B` to reverse text. The default RL deployment uses one inference GPU and one trainer GPU.

## Setup

Install the workspace environment and start the log viewer:

```bash
uv sync --all-extras --package prime-rl --package reverse-text-v1
bash scripts/tmux.sh
```

## Baseline

Start inference in one terminal:

```bash
uv run inference --model.name Qwen/Qwen3-0.6B
```

Evaluate in another:

```bash
uv run eval reverse-text-v1 --harness.id null \
  -m Qwen/Qwen3-0.6B \
  --client.base-url http://localhost:8000/v1 \
  -n 20 -r 3 \
  --sampling.max-tokens 1024 \
  --no-push
```

A representative baseline run averaged about `0.05` reward.

## SFT

The SFT config trains `PrimeIntellect/Qwen3-0.6B` on `willcb/R1-reverse-wikipedia-paragraphs-v1-1000`:

```bash
uv run sft @ examples/basic/reverse-text/sft.toml
```

For multiple GPUs, let the `sft` launcher manage the process group:

```bash
uv run sft @ examples/basic/reverse-text/sft.toml --deployment.num-gpus 2
```

The config writes final weights to `outputs/weights/step_100`. You can upload them for the RL run:

```bash
uv run hf upload "your-hf-user/Qwen3-0.6B-Reverse-Text-SFT" outputs/weights/step_100
```

## RL

Set the model to your SFT checkpoint with a CLI override when needed:

```bash
uv run rl @ examples/basic/reverse-text/rl.toml \
  --model.name "your-hf-user/Qwen3-0.6B-Reverse-Text-SFT"
```

The 20-step config writes final weights to `outputs/weights/step_20`.

## Evaluation

```bash
uv run inference --model.name "your-hf-user/Qwen3-0.6B-Reverse-Text-RL"
```

```bash
uv run eval reverse-text-v1 --harness.id null \
  -m "your-hf-user/Qwen3-0.6B-Reverse-Text-RL" \
  --client.base-url http://localhost:8000/v1 \
  -n 20 -r 3 \
  --sampling.max-tokens 1024 \
  --no-push
```

The published `PrimeIntellect/Qwen3-0.6B-Reverse-Text-RL` checkpoint averaged about `0.8` reward with this evaluation command.
