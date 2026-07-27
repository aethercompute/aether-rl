# Hendrycks Sanity

This example runs the sanity check from [Defeating the Training-Inference Mismatch](https://arxiv.org/abs/2510.26788). It trains `DeepSeek-R1-Distill-Qwen-1.5B` on MATH problems that the base model solves in 20-80% of 40 sampled attempts.

Because our trainer is asynchronous, we perform only one gradient step per batch (the inference engine generates the next batch while the trainer processes the current one).

> This example runs on 8 GPUs (4 for inference, 4 for training).

## Setup

The config uses `math-env-v1` for training and `aime24-v1` for evaluation. Install both workspace packages:

```bash
uv sync --all-extras --package prime-rl --package math-env-v1 --package aime24-v1
```

## Training

Launch locally on a node with eight GPUs:

```bash
uv run rl @ examples/basic/hendrycks-sanity/rl.toml \
  --wandb.project your-project \
  --wandb.name your-run
```
