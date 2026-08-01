# Qwen3 4B R2E-Gym smoke run

This is a bounded first agentic SWE run for `Qwen/Qwen3-4B-Instruct-2507` on
`PrimeIntellect/R2E-Gym-Subset-Verified`. Sixteen 64-rollout optimizer batches
target 1,024 accepted training rollouts. Failed or stale work and non-informative
groups can require additional generation, so 1,024 is not a hard infrastructure
request cap.

The train source excludes the `coveragepy` and `datalad` repositories. The fixed
evaluation config selects all 281 current verified rows from exactly those two
repositories, with dataset order preserved, so no repository appears in both
cohorts.

## Objective And Isolation

`r2e_gym_v1` exposes only the binary `solved` reward. It hides `/r2e_tests` during
the rollout, restores it for scoring, and captures the final patch before scoring.
GRPO has no length penalty, auxiliary reward, or post-filter; trainer KL is zero.
Tool use, patch size, response length, and partial test outcomes do not affect
reward.

The `bash` harness enables `bash` and `edit`, disables search, and runs each
rollout in a fresh Docker container. `allow = []` blocks execution-time egress
except Verifiers-managed model/tool routes. Setup may still pull the task image
and prepare the harness before the network cut. Do not replace this with the host
`subprocess` runtime.

The coordinator drops errored rollouts from training. Because GRPO requires group
scoring, any setup, sandbox, harness, finalization, or scoring error masks the
whole group rather than converting the failure to reward 0. A wrong patch that
successfully reaches the hidden-test verifier still receives the legitimate
binary reward 0.

Verifiers records the exact message graph, tool calls/results, per-turn token IDs,
sample masks, and sampling log probabilities. Aether flattens each complete
multi-turn branch without reconstructing it and stamps behavior-policy identity
on every training sample. Training batch assembly partitions by policy version
and adapter digest; `max_policy_lag = 1` drops older pending samples.

## Requirements

- Coordinator/trainer: eight CUDA GPUs with enough memory for 32,768-token LoRA
  training, persistent storage, and outbound Hugging Face/W&B access.
- Worker: one or more inference GPUs, Docker Engine with permission to start and
  remove containers, at least 32 GB host RAM for eight default execution slots,
  and enough image/cache disk for the selected R2E repositories.
- Credentials: `AETHER_COORDINATOR_TOKEN` on both roles and `WANDB_API_KEY` on
  the trainer. `HF_TOKEN` is optional for these public artifacts but recommended
  for rate limits. No search credential is used.
- Remote deployments need HTTPS in front of the coordinator. Never put secrets
  in TOML.

Reduce `execution_slots` if host RAM, file descriptors, Docker startup, or vLLM
KV cache become limiting. Keep the group size at 8.

## Validate

Run these before allocating the fleet:

```bash
export AETHER_COORDINATOR_TOKEN='validation-only'
uv run server @ examples/distributed/r2e-gym-4b/server.toml --dry-run
uv run eval @ examples/distributed/r2e-gym-4b/eval.toml \
  --dry-run --output-dir /tmp/r2e-gym-eval-validation
```

Worker preflight downloads and fingerprints the pinned model metadata and checks
the environment catalog and GPU capacity. Check Docker separately:

```bash
scripts/preflight-worker.sh @ examples/distributed/r2e-gym-4b/worker.toml
docker info
```

## Baseline Evaluation

Serve the pinned base model in one terminal:

```bash
uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tokenizer-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --served-model-name r2e-candidate \
  --max-model-len 32768
```

Run the fixed holdout in another terminal, then stop standalone vLLM before
starting the distributed worker:

```bash
uv run eval @ examples/distributed/r2e-gym-4b/eval.toml \
  --output-dir outputs/r2e-gym-4b-eval/base
```

## Train

On the coordinator/trainer host:

```bash
scripts/setup-server.sh r2e-gym-v1
export AETHER_COORDINATOR_TOKEN='<random-ascii-secret>'
export WANDB_API_KEY='<your-key>'
scripts/run-server.sh examples/distributed/r2e-gym-4b/server.toml
```

On every worker:

```bash
scripts/setup-worker.sh r2e-gym-v1
export AETHER_COORDINATOR_TOKEN='<same-random-ascii-secret>'
scripts/run-worker.sh \
  examples/distributed/r2e-gym-4b/worker.toml \
  https://coordinator.example.com \
  --state-dir "/var/lib/aether/r2e-gym-4b-$(hostname)"
```

This launches only the 16-step smoke run. Do not raise `max_steps`, task limits,
or fleet size until the stop checks below pass.

## Metrics And Stop Conditions

Start the read-only monitor on the coordinator host:

```bash
uv run monitor-report \
  --run-root outputs/qwen3-4b-r2e-gym-smoke-v1/server \
  --host 127.0.0.1 --port 8090 --refresh-seconds 10
```

Track:

| Signal | Source |
| --- | --- |
| Solved rate | Train/eval reward summaries; `solved` is the only reward component. |
| Informative groups | `inference/agg/informative_group_fraction`. |
| Turns and tokens | Monitor rollout `num_turns`, `num_output_tokens`, and `num_total_tokens`. |
| Tool errors | Raw trace tool results containing `error:` plus harness/error summaries. |
| Sandbox failures | Rollout `SandboxError` counts and coordinator recent failures. |
| Test timeouts | Scoring-stage `TaskError` timeout messages and scoring duration. |
| Patch size | UTF-8 byte length of `trace.info.patch`; inspect `patch_truncated` at the 2 MB cap. |
| Stale drops | `inference/agg/stale_drops`; also watch policy lag and pending processed rollouts. |

The `inference/agg/*` names are the existing processor counters, but the outbound
coordinator does not yet persist them into trainer JSONL or W&B. Tool-error and
patch-size aggregates likewise require a trace collector. Treat that collector
wiring as a blocker before scaling beyond this smoke run; the durable monitor and
raw result artifacts remain the source of truth meanwhile.

Stop immediately if hidden tests or grading material become visible during a
rollout, train/eval repositories overlap, a batch mixes policy identity, a failed
setup/sandbox/scoring attempt appears as reward 0, or the trainer consumes a trace
without all turns/tool spans. Also stop for sustained sandbox/test timeout growth,
stale drops above 10%, fewer than 5% informative groups after 256 clean rollouts,
all-zero or all-one solved groups over the same window, repeated tool errors,
patch truncation, context truncation, NaN/Inf, zero gradients, disk pressure, or
an unexpected rise beyond policy version 16.

## Final Evaluation

After policy version 16 is published, stop the distributed worker and serve the
final adapter:

```bash
FINAL_POLICY=$(ls -d outputs/qwen3-4b-r2e-gym-smoke-v1/server/policies/policy-v00000016-*)
uv run vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --tokenizer-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --served-model-name base \
  --max-model-len 32768 \
  --enable-lora --max-lora-rank 32 \
  --lora-modules "r2e-candidate=$FINAL_POLICY"

uv run eval @ examples/distributed/r2e-gym-4b/eval.toml \
  --output-dir outputs/r2e-gym-4b-eval/final
```

Compare all-attempt solved rate while reporting sandbox/scoring failures
separately; never reinterpret those failures as unsolved tasks.
