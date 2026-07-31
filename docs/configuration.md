# Configuration

Aether RL entrypoints use `pydantic-config`. TOML keys use `snake_case`; CLI flags use `kebab-case` and dotted nested paths. Unknown fields fail validation.

```bash
uv run server @ server.toml --port 9000
uv run worker @ worker.toml --coordinator-url https://coordinator.example.com
```

Multiple `@ file.toml` arguments compose from left to right, and later CLI values win. Boolean flags use `--flag` and `--no-flag`; dictionary overrides use JSON. Keep secrets in environment variables, not TOML.

## Server

The canonical shape is [`examples/distributed/reverse-text/server.toml`](../examples/distributed/reverse-text/server.toml).

| Field | Default | Purpose |
| --- | --- | --- |
| `run_id` | required | Immutable identifier for this run. |
| `run_root` | `server-state` | Coordinator database, spools, policies, queue, logs, and default trainer output. |
| `database_path` | `<run_root>/coordinator.sqlite` | Optional SQLite path override. |
| `trainer_config_path` | required | Trainer TOML supervised by this coordinator. |
| `trainer_output_dir` | `<run_root>/trainer` | Checkpoints, rollouts, and adapter handoff. |
| `trainer_processes` | `1` | Local `torch.distributed.run` process count. |
| `training_batch_size` | `1` | Completed groups exported in each trainer batch. |
| `published_checkpoint_keep_last` | unset | Total active/recent published full checkpoints to retain; unset retains all. |
| `host`, `port` | `127.0.0.1`, `8080` | HTTP listener behind external TLS. |
| `lease_duration_seconds` | `30` | Initial worker lease lifetime. |
| `max_policy_lag` | `0` | Allowed active-policy versions before train work is stale. |
| `result_body_limit_bytes` | 64 MiB | Maximum decompressed HTTP result body. |

The remaining timing and body-limit fields are defined by `ServerConfig` in `packages/aether-rl-configs/src/aether_rl/configs/server.py`.

`AETHER_COORDINATOR_TOKEN` is mandatory, ASCII-only, and intentionally not a config field.

### External policy distribution

The coordinator can copy each verified immutable adapter to S3-compatible storage and issue short-lived presigned locations:

```toml
[policy_distribution]
type = "s3"
bucket = "aether-policies"
prefix = "production-run"
endpoint_url = "https://ACCOUNT_ID.r2.cloudflarestorage.com"
region = "auto"
presign_ttl_seconds = 900
```

`endpoint_url` is optional for AWS S3 and must use HTTPS outside loopback. Credentials use the standard boto3 environment/configuration chain and must not be placed in TOML. Preflight verifies bucket access. Publication writes adapter files and metadata under `<prefix>/runs/<run_id>/policies/<policy_id>/`; existing objects must match their recorded size, SHA-256 metadata, and body digest. The local `run_root` remains authoritative.

## Sources

Each `[[sources]]` entry defines train or evaluation work:

```toml
[[sources]]
source_id = "reverse-text-train"
kind = "train"
environment_id = "reverse-text-v1"
environment_revision = "0.1.0"
environment_config = { taskset = { id = "reverse-text-v1" }, agent = { harness = { id = "null", runtime = { type = "subprocess" } } } }
group_size = 8
max_attempts = 3
sampling = { temperature = 1.0, max_tokens = 1024 }

[sources.algorithm]
type = "grpo"
```

Important source fields are `kind`, `group_size`, `max_attempts`, `task_limit`, `shuffle_seed`, `result_size_limit_bytes`, `assignment_timeout_seconds`, `weight`, `enabled`, and `processing_id`. `shuffle_seed` deterministically shuffles finite tasksets before applying `task_limit`; infinite tasksets require `task_limit` and cannot be shuffled. Train sampling requires a positive temperature.

Persisted scheduling fields are immutable under a given `source_id`. Algorithm, processing ID, and filter settings are loaded from current config rather than stored with the source; do not change them while pending results exist. Prefer a new run or source ID for any source change.

Supported algorithms are:

- `grpo`: group-relative reward advantages and RL loss.
- `max_rl`: mean-normalized group advantages.
- `echo`: GRPO plus selected environment-role token cross-entropy.
- `opd`: reverse-KL against an external teacher with an explicit `base_url`.

Supported rollout filters are `gibberish`, `repetition`, and `zero_advantage`. Add them with `[[sources.pre_filters]]` or `[[sources.post_filters]]`; `enforce = false` records detection metrics without removing samples.

## Base model identity

The `[base_model]` table must match on the server and all workers:

- Full model and tokenizer commit hashes.
- Canonical model-config, tokenizer, and chat-template SHA-256 digests.
- Exact vocabulary size.
- `quantization = "none"`.

Generate the table with `uv run model-identity`. Worker preflight downloads the pinned metadata and recomputes every value. The trainer's `[model].name` and `[model].revision` must match the server. If tokenizer name/revision differ from the model, set them explicitly under trainer `[tokenizer]`. If identity generation uses `--trust-remote-code`, enable the corresponding worker and trainer model settings too.

## Worker

The canonical shape is [`examples/distributed/reverse-text/worker.toml`](../examples/distributed/reverse-text/worker.toml).

| Field | Default | Purpose |
| --- | --- | --- |
| `coordinator_url` | required | HTTPS origin only, with no path or query. Loopback HTTP is allowed. |
| `state_dir` | `worker-state` | Stable identity, result spool, adapter cache, generated inference config, and inference log. |
| `execution_slots` | `1` | Maximum concurrent assignments. |
| `tensor_parallel_size` | `1` | GPUs used by local vLLM; cannot exceed visible GPUs. |
| `spool_max_entries` | `1000` | Pending result capacity; must cover all execution slots. |
| `heartbeat_interval_seconds` | `10` | Session heartbeat and lease renewal cadence. |
| `lease_wait_seconds` | `30` | Coordinator long-poll duration. |
| `inference_port` | `8000` | Loopback vLLM port. |
| `gpu_memory_utilization` | `0.9` | vLLM GPU-memory fraction. |
| `max_model_len` | unset | Optional vLLM context cap; set this for long-context models when the native maximum would exhaust worker memory. |
| `max_lora_rank` | `64` | Largest accepted adapter rank. |
| `max_loaded_policies` | `8` | Local loaded-adapter limit. |
| `adapter_cache_max_bytes` | 20 GiB | Verified adapter-cache budget. |
| `result_compression` | `zstd` | Compress MessagePack result uploads; `identity` disables compression. |
| `result_upload_concurrency` | `2` | Concurrent durable-spool upload tasks, from 1 through 16. |
| `policy_download_allowed_origins` | `[]` | Exact approved HTTPS origins for presigned adapter URLs. |
| `shardcast_servers` | `[]` | Optional HTTPS SHARDCAST relay base URLs, tried first for safetensors. |
| `shardcast_download_concurrency` | `4` | Concurrent SHARDCAST shard requests. |
| `policy_download_attempts` | `3` | Verified download attempts with persistent HTTP Range resume. |
| `policy_coordinator_fallback` | `true` | Fall back to authenticated coordinator file delivery. |
| `policy_prefetch_interval_seconds` | `5` | Active-policy disk-cache polling interval; unset disables prefetch. |

Each sorted, unique `[[environments]]` entry contains `id`, `package`, `revision`, and verifier `config`. The package must be installed, its version must equal `revision`, and its resolved environment ID must match `id`. Workers enforce package versions; operators must keep the coordinator's installed taskset package aligned with the advertised source revision.

The worker generates `<state_dir>/inference.toml`; do not maintain that file manually. Configure the distributed inference subset through worker fields.

External origins are exact scheme/authority matches and must use HTTPS. Coordinator authorization is never sent to these origins. Download order is SHARDCAST for `adapter_model.safetensors`, an approved presigned location, then the coordinator when fallback is enabled. SHARDCAST does not carry `adapter_config.json`, so disabling coordinator fallback also requires an approved presigned origin. Prefetch verifies the adapter into the disk cache; vLLM loads it only when an assignment uses it. Result limits in server and source configuration apply after zstd decompression.

## Policy relay

The optional relay downloads the current verified adapter and exposes SHARDCAST through an HTTPS proxy:

```toml
coordinator_url = "https://coordinator.example.com"
state_dir = "policy-relay-state"
port = 8000
poll_interval_seconds = 2
request_timeout_seconds = 300
max_versions = 8
shard_size_bytes = 8388608
policy_download_allowed_origins = ["https://ACCOUNT_ID.r2.cloudflarestorage.com"]
```

`AETHER_COORDINATOR_TOKEN` is required. Run `scripts/setup-relay.sh`, then `scripts/run-relay.sh policy-relay.toml`. Put TLS in front of the relay and configure that HTTPS URL in worker `shardcast_servers`; do not expose its plain HTTP listener directly.

## Trainer

The coordinator owns trainer launch and resume. The canonical shape is [`examples/distributed/reverse-text/trainer.toml`](../examples/distributed/reverse-text/trainer.toml).

Distributed training requires:

- `[model.lora]` with no `modules_to_save`.
- `[ckpt] interval = 1` with complete optimizer, scheduler, dataloader, and progress state.
- No trainer-owned checkpoint pruning, custom checkpoint output directory, `weights_only`, skip flags, or configured `resume_step`.
- Filesystem weight broadcast in safetensors format.
- Exactly one run.

The coordinator sets the trainer output directory and resume step. Common trainer sections are `[model]`, `[model.lora]`, `[optim]`, `[scheduler]`, `[loss]`, `[ckpt]`, `[wandb]`, `[file_monitor]`, and `[metrics_server]`.

Set server `published_checkpoint_keep_last` to bound full-checkpoint retention. Cleanup runs only after durable policy activation, never removes unpublished future checkpoints, and also removes the now-redundant trainer broadcast copy. Immutable published policy adapters remain retained. Lowering this value applies on the next publication or coordinator startup and irreversibly removes newly eligible checkpoints, so back up the stopped run first when rollback history matters. Leave trainer `ckpt.keep_last` and `ckpt.keep_interval` unset.

Under `[wandb]`, set `log_metrics` to an exact metric-name allowlist to avoid auto-generating panels for every trainer diagnostic. Set `create_overview = false` when the generic project workspace is not useful for the run.

Do not change trainer topology, LoRA shape, optimizer, or model identity when restarting an existing run unless checkpoint compatibility has been independently established.
