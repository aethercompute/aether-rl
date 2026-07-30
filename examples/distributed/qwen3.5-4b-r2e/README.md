# Distributed Qwen3.5 4B R2E-Gym pilot

This run trains `Qwen/Qwen3.5-4B` to resolve repository issues through
mini-swe-agent. Rollouts execute in per-task local Docker containers, and
`r2e-gym-v1` assigns a binary reward by restoring hidden tests and matching their
outcomes to the gold solution. The coordinator uses the 4,522-row
`PrimeIntellect/R2E-Gym-Subset-Verified` derivative of
`R2E-Gym/R2E-Gym-Subset`; its excluded 56 rows do not pass gold-patch validation.

The checked-in configuration is a 50-step consumer-hardware pilot. It is intended
to establish base pass rate, useful task strata, token lengths, sandbox throughput,
and failure rates before committing to a larger run.

## Training design

- Commit hashes ending in `0` form a deterministic holdout of approximately 1/16
  of the dataset. They never enter the training source.
- Training uses MaxRL with eight rollouts per task. Binary-reward groups with no
  advantage are discarded before batching.
- Qwen's precise-coding sampling settings are used: temperature 0.6, top-p 0.95,
  and top-k 20. Each turn is capped at 4,096 generated tokens.
- The framework caps a rollout at 50 model turns or 24,576 total trace tokens.
  vLLM and the trainer use a 32,768-token limit so late rewarded edits are retained.
- Rank-64 LoRA covers full-attention, DeltaNet, and MLP projections.
- The mini-swe-agent release is pinned to 2.4.5 and uses its minimal `mini`
  profile. Model calls still pass through Verifiers interception, preserving exact
  sampled token IDs and log probabilities for training.

## Capacity

Start with one worker process per rollout GPU, tensor parallelism 1, and two Docker
execution slots per host. Increase slots one at a time only after observing CPU,
RAM, Docker disk, image-pull latency, and vLLM scheduling. Each task requests four
CPU cores and 4 GB RAM; concurrent test suites can be more limiting than inference.

The Qwen checkpoint is approximately 9.3 GB in BF16 before KV cache and adapters.
The trainer writes a complete resumable checkpoint every step. Keep the run root
and Docker data root on filesystems with ample free space, and measure checkpoint,
policy, trace, and image growth during the first few steps.

## Setup

Use the same repository revision on all machines. The worker user must be able to
run `docker version` without elevation. Docker images are pulled from the task rows,
so plan for substantial local image storage and registry traffic.

On the coordinator/trainer host:

```bash
scripts/setup-server.sh r2e-gym-v1
export AETHER_COORDINATOR_TOKEN='<random-ascii-secret>'
export WANDB_API_KEY='<your-key>'
```

On every rollout worker:

```bash
scripts/setup-worker.sh r2e-gym-v1
export AETHER_COORDINATOR_TOKEN='<same-random-ascii-secret>'
docker version
```

The model identity in the configs was generated for revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` with:

```bash
uv run model-identity \
  --model-name Qwen/Qwen3.5-4B \
  --model-revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
```

## Preflight

Before the distributed run, validate representative images and the reward path.
A no-op should score zero and the reconstructed gold patch should score one. Also
run a small base-policy sample across repositories and task sizes; a useful MaxRL
stratum needs both successes and failures within groups.

Validate both role configurations without starting services:

```bash
scripts/preflight-server.sh \
  @ examples/distributed/qwen3.5-4b-r2e/server.toml
scripts/preflight-worker.sh \
  @ examples/distributed/qwen3.5-4b-r2e/worker.toml
```

Before scaling out, complete a one-step end-to-end run and verify that vLLM loads
the published adapter, including the DeltaNet projection targets.

## Launch

Start the coordinator and supervised trainer:

```bash
scripts/run-server.sh examples/distributed/qwen3.5-4b-r2e/server.toml
```

Wait for `/ready`, then start each worker with a unique persistent state directory:

```bash
scripts/run-worker.sh \
  examples/distributed/qwen3.5-4b-r2e/worker.toml \
  https://coordinator.example.com \
  --state-dir "/var/lib/aether/qwen3p5-r2e-$(hostname)"
```

Use an HTTPS proxy, VPN, or secure mesh for remote workers. Preserve authorization
and protocol headers, long polling, and 64 MiB result bodies.

## Monitor and scale

```bash
uv run --no-default-groups monitor-report \
  --run-root outputs/qwen3p5-4b-r2e-pilot-v1/server \
  --host 127.0.0.1 \
  --port 8090 \
  --refresh-seconds 10

uv run --no-default-groups eval-report \
  --run-root outputs/qwen3p5-4b-r2e-pilot-v1/server \
  --source-id qwen3p5-4b-r2e-holdout-v1 \
  --watch-seconds 20
```

Compare holdout reward, error rate, turns, token lengths, truncation, and wall time
against policy 0. Inspect all-zero group frequency: if it is high, define weighted
training sources over complexity bands that have measurable base successes rather
than increasing steps on uninformative tasks. Raise the 32K context only when
truncation affects successful trajectories and both trainer memory and worker KV
cache have measured headroom.

Create a new `run_id` and run directory when changing source definitions, model
identity, LoRA shape, trainer topology, or context length after state exists.
