# Alphabet Sort

This example trains `Qwen3-4B-Instruct-2507` with LoRA on the multi-turn `alphabet-sort-v1` taskset. It proceeds directly to RL without an SFT warmup. The default RL launcher assigns separate inference and trainer GPUs.

## Setup

```bash
uv sync --all-extras --package prime-rl --package alphabet-sort-v1
bash scripts/tmux.sh
```

The task asks the model to maintain a cumulative list sorted by first or last name and mark newly added names. The RL config uses LoRA rank 32 and alpha 64.

## Baseline

```bash
uv run inference --enable-lora --model.name Qwen/Qwen3-4B-Instruct-2507
```

```bash
uv run eval alphabet-sort-v1 --harness.id null \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  --client.base-url http://localhost:8000/v1 \
  -n 20 -r 3 \
  --sampling.max-tokens 768 \
  --args '{"min_turns": 3, "max_turns": 3, "min_names_per_turn": 1, "max_names_per_turn": 4, "task": {"similarity_power": 8, "power_per_turn": false}}' \
  --no-push
```

A representative baseline averaged about `0.26` reward with no perfect attempts.

## RL

```bash
uv run rl @ examples/basic/alphabet-sort/rl.toml
```

The 200-step config writes final weights to `outputs/weights/step_200`:

```bash
uv run hf upload "your-hf-user/Qwen3-4B-Instruct-AlphabetSort-RL" outputs/weights/step_200
```

## Evaluation

```bash
uv run inference --enable-lora --model.name "your-hf-user/Qwen3-4B-Instruct-AlphabetSort-RL"
```

```bash
uv run eval alphabet-sort-v1 --harness.id null \
  -m "your-hf-user/Qwen3-4B-Instruct-AlphabetSort-RL" \
  --client.base-url http://localhost:8000/v1 \
  -n 20 -r 3 \
  --sampling.max-tokens 768 \
  --args '{"min_turns": 3, "max_turns": 3, "min_names_per_turn": 1, "max_names_per_turn": 4, "task": {"similarity_power": 8, "power_per_turn": false}}' \
  --no-push
```

The published `PrimeIntellect/Qwen3-4B-Instruct-AlphabetSort-RL` checkpoint averaged about `0.81` reward in this evaluation.
