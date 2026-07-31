# Efficient DeepSeek R1 DAPO math run

This short run trains `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` for ten optimizer
steps on the pinned English-only `open-r1/DAPO-Math-17k-Processed` subset. It uses
eight rollouts per problem and eight complete groups per step, for 640 train
rollouts. Failed groups may require replacement generation. The 128-task
deterministic holdout is disjoint after lightweight prompt normalization.

## Objective

The environment gives binary correctness only for a closed reasoning response
whose only non-whitespace final content is `Answer: \boxed{<integer>}`. Every
published English DAPO gold is prevalidated as an integer, so broad symbolic
normalization is avoided.

The `grpo` algorithm mean-centers rewards without standard-deviation normalization.
The trainer additionally uses Dr. GRPO's fixed 16,384-token response denominator
and no KL term. Before centering, correct rollouts longer than the shortest correct
thinking trace in their group receive a flat `0.5` penalty:

```text
wrong                         -> 0.0
shortest correct              -> 1.0
other correct, longer thought -> 0.5
```

Only sampled tokens before `</think>` enter `thinking_tokens`. Prompt tokens, the
closing delimiter, boxed answer, EOS, and padding do not affect the length penalty.
The flat penalty is intentionally strong while keeping every correct rollout above
every incorrect rollout.

## Limits

Completions may use up to 16,384 tokens. Worker and trainer context are both 20,480
tokens so the final answer is retained rather than silently truncated. The run is
short by step and rollout count, but DeepSeek reasoning can still make wall time
substantial; start with four worker execution slots and adjust only after measuring
KV-cache utilization and generation throughput.

## Baseline evaluation

Run the exact holdout evaluation before training. In one terminal, start the
token-aware vLLM endpoint for the pinned base model:

```bash
uv run vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --tokenizer-revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --served-model-name dapo-candidate \
  --max-model-len 20480
```

In a second terminal:

```bash
uv run eval @ examples/distributed/dapo-math-1.5b/eval.toml \
  --output-dir outputs/dapo-math-1.5b-eval/base
```

The live eval dashboard aggregates every reward and metric key, including strict
correctness, `format_valid`, and `thinking_tokens`. Record those means, then stop
this standalone vLLM process before launching the rollout worker.

## Train

On the coordinator/trainer host:

```bash
scripts/setup-server.sh dapo-math-v1
export AETHER_COORDINATOR_TOKEN='<random-ascii-secret>'
export WANDB_API_KEY='<your-key>'
scripts/run-server.sh examples/distributed/dapo-math-1.5b/server.toml
```

On every worker:

```bash
scripts/setup-worker.sh dapo-math-v1
export AETHER_COORDINATOR_TOKEN='<same-random-ascii-secret>'
scripts/run-worker.sh \
  examples/distributed/dapo-math-1.5b/worker.toml \
  https://coordinator.example.com \
  --state-dir "/var/lib/aether/dapo-math-$(hostname)"
```

## Final evaluation

After policy version 10 is published, stop the distributed worker. In one terminal,
serve the final adapter from the coordinator host:

```bash
FINAL_POLICY=$(ls -d outputs/deepseek-r1-1p5b-dapo-efficient-v1/server/policies/policy-v00000010-*)
uv run vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --tokenizer-revision ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
  --served-model-name base \
  --max-model-len 20480 \
  --enable-lora \
  --lora-modules "dapo-candidate=$FINAL_POLICY"
```

In a second terminal:

```bash
uv run eval @ examples/distributed/dapo-math-1.5b/eval.toml \
  --output-dir outputs/dapo-math-1.5b-eval/final
```

Compare the same 128 tasks in `base` and `final`. A successful run improves strict
correctness while reducing `thinking_tokens`; format improvement alone is not
evidence of better math.
