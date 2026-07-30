# Distributed DeepScaleR 1.5B stage 1

This run trains `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` on
`agentica-org/DeepScaleR-Preview-Dataset` with GRPO, binary boxed-answer rewards,
and outbound rollout workers. The coordinator and single-GPU LoRA trainer are intended
for the RTX 3090 machine; each RTX 4090 or RTX 5090 machine runs an independent worker
and local vLLM instance.

The checked-in stage is a first consumer-hardware run, not an exact reproduction of the
8-A100 DeepScaleR phase. It performs 50 optimizer steps. Each step consumes 128 rollouts
(16 problems with 8 samples each), for up to 6,400 train rollouts. Training completions
use temperature 0.6, top-p 0.95, and a 24,576-token cap. AIME 2024 greedy evaluations use
a 32,768-token cap and run alongside training at low scheduler weight.

## Format and identity

The model, tokenizer, and dataset are pinned to immutable Hugging Face revisions. The
current official DeepSeek tokenizer enforces the model card's recommended `<think>`
prefill. Rendering is therefore:

```text
<｜begin▁of▁sentence｜><｜User｜>{problem} Let's think step by step and output the final answer within \boxed{}.<｜Assistant｜><think>
```

There is no system message. The opening `<think>` is prompt scaffold and is not a sampled
training token. The policy generates the reasoning, `</think>`, and final content with its
answer in `\boxed{...}`. Aether stores the generated reasoning and final answer separately
as `reasoning_content` and `content`, while retaining all generated tokens in the training
sample. Do not stop generation at `</think>`; EOS or the token cap must end the rollout.
The dataset's worked `solution` field is not used. Six rows with empty answers are excluded
and two malformed gold strings are repaired. Scoring is local and deterministic, with no
reference-model judge or API key.

## Capacity

The trainer uses a 28,672-token training sequence limit so a 24,576-token completion and
the longest rendered dataset prompt fit without truncating the rewarded final answer. It
enables the trainer defaults for activation checkpointing, activation offload, and
optimizer-state CPU offload. Start with at least 64 GiB of system RAM on the trainer host.

Each full checkpoint is expected to use about 7.5 GiB. The coordinator keeps the active
checkpoint and two prior checkpoints, and removes each redundant trainer broadcast after
its immutable policy is activated. Published rank-64 policy adapters still use roughly
14 GiB across 50 steps, and long rollout records also accumulate. Provision at least
200 GiB on the filesystem containing `run_root`, then measure actual checkpoint and trace
growth during the first few steps. The original 1,040-step phase still needs substantially
more policy, rollout, and operational storage as well as a much larger rollout fleet.

## Setup

Use the same repository revision on every machine. On the 3090 trainer/coordinator:

```bash
scripts/setup-server.sh deepscaler-v1
export AETHER_COORDINATOR_TOKEN='<random-ascii-secret>'
export WANDB_API_KEY='<your-key>'
```

On every rollout worker:

```bash
scripts/setup-worker.sh deepscaler-v1
export AETHER_COORDINATOR_TOKEN='<same-random-ascii-secret>'
```

The coordinator binds to loopback. Put an HTTPS reverse proxy, VPN gateway, or secure mesh
in front of it for remote Vast workers; do not expose port 8080 directly. The proxy must
preserve `Authorization` and `Aether-Protocol-Version`, support long polling, and permit
64 MiB decompressed request bodies. It must preserve `Content-Encoding: zstd` and HTTP
Range headers used by resumable policy downloads.

For optional R2 delivery, replace the placeholders in `server-r2.toml`, export standard
boto3-compatible R2 credentials on the coordinator, and compose it after the base config:

```bash
scripts/run-server.sh examples/distributed/deepscaler-1.5b/server.toml \
  @ examples/distributed/deepscaler-1.5b/server-r2.toml
```

The baseline worker config still uses coordinator delivery. To use R2 and an optional
SHARDCAST endpoint, replace the origins in `worker-external-policy.toml` and compose it
after `worker.toml`. Coordinator authorization is never forwarded to either endpoint.

## Launch

Start the coordinator and supervised trainer on the 3090 machine:

```bash
scripts/run-server.sh examples/distributed/deepscaler-1.5b/server.toml
```

Wait for `/ready`, then start one worker on each rollout machine. Every worker needs a
unique persistent state directory:

```bash
scripts/run-worker.sh \
  examples/distributed/deepscaler-1.5b/worker.toml \
  https://coordinator.example.com \
  --state-dir "/var/lib/aether/deepscaler-$(hostname)"
```

To run the optional SHARDCAST bridge, replace placeholders in `policy-relay.toml`, put
HTTPS in front of its port, then start it before workers:

```bash
scripts/setup-relay.sh
scripts/run-relay.sh examples/distributed/deepscaler-1.5b/policy-relay.toml
```

Workers configured with `worker-external-policy.toml` try SHARDCAST, then R2, then the
coordinator. Every path is size- and SHA-256-verified; the coordinator remains the source
of truth and fallback.

The default 16 execution slots are conservative for either GPU. Increase one machine at a
time after watching GPU memory and generation throughput, for example
`--execution-slots 24` on a 5090. Keep `tensor_parallel_size = 1`; run one worker process
per single-GPU Vast instance. A unique `state_dir` is mandatory, and a second worker on
the same host also needs a different `--inference-port`.

## Monitor

```bash
curl -fsS https://coordinator.example.com/ready
curl -fsS https://coordinator.example.com/api/v1/status \
  -H "Authorization: Bearer $AETHER_COORDINATOR_TOKEN" \
  -H "Aether-Protocol-Version: 1"

uv run --no-default-groups eval-report \
  --run-root outputs/deepscaler-r1-qwen-1p5b-stage1-v3/server \
  --source-id deepscaler-stage1-v3-aime24-eval \
  --watch-seconds 20

uv run --no-default-groups monitor-report \
  --run-root outputs/deepscaler-r1-qwen-1p5b-stage1-v3/server \
  --host 127.0.0.1 \
  --port 8090 \
  --refresh-seconds 10
```

Open `http://127.0.0.1:8090` on the coordinator host to watch the plain HTML monitor. It
reads the run directory and shows coordinator state, workers, rollout speed windows,
trainer metrics, decoded train/eval reward and format metrics, token lengths, truncation,
recent rollouts, and failures. Policy 0 is the untouched base model. The final policy
normally receives no new evaluation leases, so compare earlier policy versions as well as
the terminal checkpoint. This single-sample greedy AIME series is a stable health metric,
not directly comparable to DeepScaleR's published multi-sample evaluation at temperature
0.6.

## Scaling changes

Make throughput-only worker changes with CLI overrides. Before the first launch, the main
consumer-scale knobs are `training_batch_size`, `group_size`, `max_tokens`, `seq_len`, and
`max_steps`. Do not change source definitions, model/tokenizer identity, LoRA shape, or
trainer topology after state exists under this `run_id`; create a new run directory and
`run_id` for a different experiment.

The original phase used 1,024 rollouts per update (128 problems times 8) and 1,040 steps
with full-parameter training at learning rate `1e-6`. This config uses rank-64 LoRA, a
256-rollout update, and learning rate `1e-5` to fit the available trainer and rollout
fleet. It keeps the published prompt, temperature, top-p, GRPO group size, clip thresholds,
gradient clipping, and KL coefficient while using the published final-phase 24K train cap.
