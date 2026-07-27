# Wordle

This example applies an SFT warmup and multi-turn RL to `Qwen3-1.7B`. The shipped RL config allocates six inference GPUs and two trainer GPUs.

## Setup

```bash
uv sync --all-extras --package prime-rl --package wordle-v1
bash scripts/tmux.sh
```

## Baseline

Start inference:

```bash
uv run inference --model.name Qwen/Qwen3-1.7B
```

Run the held-out evaluation:

```bash
uv run eval wordle-v1 --harness.id null \
  -m Qwen/Qwen3-1.7B \
  --client.base-url http://localhost:8000/v1 \
  -n 20 -r 3 \
  --sampling.max-tokens 1024 \
  --no-push
```

A representative baseline run averaged about `0.2` reward and had a `0%` win rate.

## SFT

The SFT config trains `PrimeIntellect/Qwen3-1.7B` on `willcb/V3-wordle`:

```bash
uv run sft @ examples/basic/wordle/sft.toml
```

For multiple GPUs:

```bash
uv run sft @ examples/basic/wordle/sft.toml --deployment.num-gpus 2
```

The config writes final weights to `outputs/weights/step_20`:

```bash
uv run hf upload "your-hf-user/Qwen3-1.7B-Wordle-SFT" outputs/weights/step_20
```

## RL

```bash
uv run rl @ examples/basic/wordle/rl.toml \
  --model.name "your-hf-user/Qwen3-1.7B-Wordle-SFT"
```

The 200-step config writes final weights to `outputs/weights/step_200`.

## Evaluation

```bash
uv run inference --model.name "your-hf-user/Qwen3-1.7B-Wordle-RL"
```

```bash
uv run eval wordle-v1 --harness.id null \
  -m "your-hf-user/Qwen3-1.7B-Wordle-RL" \
  --client.base-url http://localhost:8000/v1 \
  -n 20 -r 3 \
  --sampling.max-tokens 1024 \
  --no-push
```

The published `PrimeIntellect/Qwen3-1.7B-Wordle-RL` checkpoint reached about a `60%` win rate in this evaluation.
