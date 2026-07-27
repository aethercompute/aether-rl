# Inference

This page covers vLLM configuration, deployment shapes, routing, and KV-cache offload.

## Table of Contents

- [Overview](#overview)
- [Single-Node](#single-node)
- [Multi-Node](#multi-node)
- [P/D Disaggregation](#pd-disaggregation)
- [Router](#router)
  - [Routing policies](#routing-policies)
- [Advanced Configuration](#advanced-configuration)
  - [KV Cache Offload](#kv-cache-offload)
  - [Optimized P/D disaggregation deployment](#optimized-pd-disaggregation-deployment)
  - [Other vLLM features](#other-vllm-features)
  - [Router Replay](#router-replay)


## Overview

`prime-rl` uses vLLM for policy inference and exposes its configuration through `InferenceConfig`.

Three deployment shapes are available:

- [Single-node](#single-node): one local vLLM deployment; this is the default.
- [Multi-node](#multi-node): independent node-local replicas behind a router.
- [Disaggregated](#pd-disaggregation): separate prefill and decode replicas with KV transfer.

Select the deployment shape with `InferenceDeploymentConfig`. The following snippets are nested inside an RL config; omit the `inference.` prefix for the standalone `uv run inference` entrypoint.

```toml
[inference.deployment]
type = "single_node" # or "multi_node" or "disaggregated"
```

Model settings are nested under `model`:

```toml
[inference.model]
name = "PrimeIntellect/INTELLECT-3"
max_model_len = 32768
```

## Single-Node

Single-node inference is the default and is suitable for local development and models that fit within one node.

```toml
[inference.deployment]
type = "single_node"
```

```toml
[inference]
enable_expert_parallel = true # defaults to False

[inference.parallel]
tp = 2
dp = 4

[inference.deployment]
type = "single_node"
```

Higher DP generally favors throughput, while higher TP can reduce per-request latency and model memory per GPU. Size both the model and KV cache for the available GPU memory. For MoE models, expert parallelism can also shard expert weights.

You can also increase the available KV cache memory by enabling `inference.kv_cache_offload`. More details in the [Advanced Configuration](#advanced-configuration) section.


## Multi-Node

Multi-node inference runs an independent vLLM deployment on each node and routes requests across their endpoints. Parallelism configuration applies within each node; set `inference.deployment.num_nodes` to the replica count.

```toml
[inference.deployment]
type = "multi_node"
num_nodes = 2

[inference.model]
name = "PrimeIntellect/INTELLECT-3"

[inference.parallel]
tp = 2
dp = 4
```

With eight GPUs per node, this starts four routed TP=2 endpoints on each node. Without expert parallelism, each endpoint is an independent engine. With expert parallelism enabled, the four endpoints on a node form a node-local DP/EP group; EP never spans nodes. Select `vllm-router` or `llm-d` with `[...deployment.router]`.

## P/D Disaggregation

P/D disaggregation runs prefill and decode on separate replicas and transfers KV-cache state between them. It is intended for workloads where independent scaling of the two stages improves latency or throughput.

This deployment shape is defined by setting `inference.deployment.type = "disaggregated"` and choosing how many nodes each prefill and decode replica spans.

```toml
[inference.deployment]
type = "disaggregated"
prefill_nodes_per_replica = 2
decode_nodes_per_replica = 2
```

Set `num_prefill_replicas` and `num_decode_replicas` to run multiple instances of either role.

```toml
[inference.deployment]
type = "disaggregated"
prefill_nodes_per_replica = 2
num_prefill_replicas = 2
decode_nodes_per_replica = 2
num_decode_replicas = 1
```

Now each prefill replica spans 2 nodes and each decode replica spans 2 nodes. With 2 prefill replicas and 1 decode replica, one inference island spans 6 nodes.

For RL runs, the top-level deployment can multiply that whole inference island by setting `deployment.num_infer_replicas`. `deployment.num_infer_nodes` is inferred from the nested inference deployment when you omit it.

```toml
[deployment] # top-level RL deployment
type = "multi_node"
num_train_nodes = 4

num_infer_replicas = 3
```

This will run 3 inference islands, each running on 6 nodes. The total inference deployment will span 18 nodes and start 3 separate router instances.


## Router

Multi-node and disaggregated deployments front their vLLM backends with a router, configured via a discriminated `[...deployment.router]` block (`type = "vllm-router" | "llm-d"`):

```toml
[inference.deployment.router]   # or [deployment.router] for the standalone inference entrypoint
type = "llm-d"                  # "vllm-router" (default) or "llm-d"
# llm-d-only knobs (all optional):
scorers = { "prefix-cache-scorer" = 3.0, "active-request-scorer" = 2.0 }   # base, applied to every profile
prefill_scorer_overrides = { "queue-scorer" = 2.0, "kv-cache-utilization-scorer" = 2.0 }  # merged onto the P/D prefill profile
decode_scorer_overrides = {}    # merged onto the P/D decode profile
non_cached_tokens = 16          # below this many non-cached prompt tokens, skip remote prefill (P/D)
```

- **`vllm-router`** (default) — our fork of [vllm-router](https://github.com/PrimeIntellect-ai/router). Knob: `policy`.
- **`llm-d`** — the upstream [llm-d](https://llm-d.ai) Endpoint Picker (EPP) + Envoy proxy. Routing combines **prefix-cache affinity** (grouped rollouts reuse a cached prefix and skip prefill) with the **`active-request-scorer`** — an in-flight load balancer that spreads requests across ranks immediately, unlike the metrics-scraped `queue-scorer` / `kv-cache-utilization-scorer` / `load-aware-scorer` (which lag and concentrate bursts of same-prefix requests). The scorer weights follow the upstream llm-d P/D guide; tune via `scorers` (base) + `prefill_scorer_overrides` / `decode_scorer_overrides` (per-profile, P/D). Does not support `enable_return_routed_experts` (router replay).

Both backends support cache-aware request routing and P/D disaggregation.

### Routing policies

`consistent_hash` is the default `vllm-router` policy. It hashes a request header to preserve KV-cache locality across turns. Configure the state field used for that header with `orchestrator.model.client.extra_headers_from_state`:

```toml
[orchestrator.model.client.extra_headers_from_state]
X-Session-ID = "trajectory_id" # this is the default - each rollout has a unique trajectory_id and router expects X-Session-ID
```

Use `round_robin` when load distribution matters more than session affinity.


## Advanced Configuration

### KV Cache Offload

Maximizing KV-Cache space is crucial to support high-concurrency workloads. You can offload the KV cache to CPU memory (and, behind it, disk) by setting `inference.kv_cache_offload`. It is a discriminated config with two composable tiers, `cpu` and `disk`: a `cpu` tier is always required, and an optional `disk` tier is layered behind it (GPU → DRAM → disk). Disk-only is not supported.

The `type` field selects the backend:

- `native` — vLLM's built-in offloading. CPU-only uses `OffloadingConnector`; CPU+disk uses `TieringOffloadingSpec` (a CPU primary tier with a filesystem secondary tier). Fully self-contained — no extra processes.
- `mooncake` — a [Mooncake](https://github.com/kvcache-ai/Mooncake) **shared distributed store** (SLURM only). One `mooncake_master` + metadata server runs on the head inference node; every inference node runs a `mooncake_client` that contributes its DRAM (and, with `disk`, SSD) segment to that *single* pool. Because blocks are keyed by model + parallel rank + content hash (no instance id), a prefix cached by one node/replica is reusable by all of them over RDMA — pooling every node's CPU RAM into one KV cache. Use `native` for local/single-process runs.

Native CPU offload:

```toml
[inference.kv_cache_offload]
type = "native"
[inference.kv_cache_offload.cpu]
num_bytes = 128_000_000_000
```

Native CPU and disk tiering:

```toml
[inference.kv_cache_offload]
type = "native"
[inference.kv_cache_offload.cpu]
num_bytes = 128_000_000_000
[inference.kv_cache_offload.disk]
path = "/scratch/kv"
```

Mooncake CPU and disk tiering:

```toml
[inference.kv_cache_offload]
type = "mooncake"
[inference.kv_cache_offload.cpu]
num_bytes = 128_000_000_000
[inference.kv_cache_offload.disk]
path = "/scratch/kv"
```

For `native`, `cpu.num_bytes` is the aggregate CPU KV pool for the instance (vLLM shards it across workers). For `mooncake`, `cpu.num_bytes` is the DRAM each node contributes to the shared pool (so the total pool ≈ `num_bytes × #inference-nodes`); the store uses RDMA, so it requires an RDMA-capable fabric. Enabling offload automatically enables prefix caching.


### Optimized P/D disaggregation deployment

P/D launch sets the decode `all2all_backend` to `deepep_low_latency` and the prefill backend to `deepep_high_throughput`. Override role-specific vLLM options with `prefill_vllm_overrides` and `decode_vllm_overrides` under `[inference.deployment]`.

KV-cache transfer uses the NIXL connector.

> **Required:** Build NIXL against UCX 1.19.x from source for prefill-to-decode KV transfer. See [Disaggregated Prefill/Decode Inference](advanced.md#disaggregated-prefilldecode-inference).

Set role-specific environment variables with `prefill_env_vars` and `decode_env_vars`:

```toml
[inference.deployment]
type = "disaggregated"

prefill_env_vars = {"VLLM_ENABLE_MOE_DP_CHUNK"="0", "VLLM_DEEP_GEMM_WARMUP"="skip"}
decode_env_vars = {"VLLM_DEEP_GEMM_WARMUP"="skip"}
```

These are role-specific and layer on top of [`env_vars`](configuration.md#environment-variables) shared by all inference processes regardless of role.

### Other vLLM features

Fields such as `enable_dbo` and `enable_eplb` are exposed directly. Pass other vLLM options through `inference.vllm_extra`:

```toml
[inference.vllm_extra]
headless = true
```

### Router Replay

Router replay sends inference-time expert routing decisions to the trainer instead of recomputing them.

To enable router replay, you can set `inference.enable_return_routed_experts = true`.

```toml
[trainer]
enable_router_replay = true # this will also auto-set the inference.enable_return_routed_experts = true

[inference]
enable_return_routed_experts = true
```

Router replay increases HTTP payload size. If request handling becomes a bottleneck, increase the worker count in the affected environment's `pool` configuration.

Router replay is incompatible with CPU KV-cache offload.
