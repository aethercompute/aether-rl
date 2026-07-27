# Wiki Search

This example trains `Qwen3-4B-Instruct-2507` with LoRA to answer trivia questions using Wikipedia search tools. The shipped RL config allocates six inference GPUs and two trainer GPUs.

The taskset provides tools to search page titles, list page sections, and read a section. It stores its ChromaDB index under `~/.cache/wiki_search` by default and uses a configured reference-answer judge for scoring.

## Setup

```bash
uv sync --all-extras --package prime-rl --package wiki-search-v1
export OPENAI_API_KEY="your-api-key"
bash scripts/tmux.sh
```

The first run builds the local index from `willcb/rare-wiki-pages`. Set `WIKI_SEARCH_CACHE` to use another cache directory.

## Baseline

```bash
uv run inference \
  --enable-lora \
  --model.name Qwen/Qwen3-4B-Instruct-2507 \
  --model.tool-call-parser hermes
```

```bash
uv run eval wiki-search-v1 --harness.id null \
  -m Qwen/Qwen3-4B-Instruct-2507 \
  --client.base-url http://localhost:8000/v1 \
  -n 20 -r 3 \
  --sampling.max-tokens 512 \
  --no-push
```

## RL

The unified config contains the trainer, orchestrator, inference, LoRA, tool parser, and online difficulty-buffer settings:

```bash
uv run rl @ examples/basic/wiki-search/rl.toml
```

The 200-step config writes final weights to `outputs/weights/step_200`:

```bash
uv run hf upload "your-hf-user/Qwen3-4B-Instruct-WikiSearch-RL" outputs/weights/step_200
```

## Evaluation

```bash
uv run inference \
  --enable-lora \
  --model.name "your-hf-user/Qwen3-4B-Instruct-WikiSearch-RL" \
  --model.tool-call-parser hermes
```

```bash
uv run eval wiki-search-v1 --harness.id null \
  -m "your-hf-user/Qwen3-4B-Instruct-WikiSearch-RL" \
  --client.base-url http://localhost:8000/v1 \
  -n 20 -r 3 \
  --sampling.max-tokens 512 \
  --no-push
```

To replace the reference judge in an RL overlay, keep the environment configuration nested under `env`:

```toml
[[orchestrator.train.env]]
name = "wiki-search"
env.taskset = { id = "wiki-search-v1", task = { judges = [{ id = "reference", model = "openai/gpt-5.4-nano" }] } }
env.agent.harness = { id = "null", runtime = { type = "subprocess" } }
```
