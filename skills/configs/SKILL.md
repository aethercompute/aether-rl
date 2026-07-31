---
name: configs
description: Configure Aether RL server, worker, trainer, sources, and CLI overrides.
---

# Configs

Aether RL uses `pydantic-config`. Entrypoints accept one or more files via `@ path.toml`; later files and CLI values override earlier values.

```bash
uv run server @ server.toml --port 9000
uv run worker @ worker.toml --coordinator-url https://coordinator.example.com
```

- TOML uses snake_case; CLI flags use kebab-case.
- Nested CLI fields use dotted paths.
- Booleans use `--flag` and `--no-flag`.
- Dict CLI values use JSON.
- Discriminated unions select variants through `type`.
- Unknown fields are errors.
- Secrets belong in environment variables, never TOML.

Run configurations are workload-specific and are not checked in. Generate the shared model identity block for `server.toml` and `worker.toml` with:

```bash
uv run model-identity --model-name 'organization/model' --model-revision '<40-character-commit>'
```

The server and all workers require the same full model/tokenizer revisions, fingerprints, vocabulary size, and quantization identity. The trainer model name/revision must match; configure a distinct tokenizer and remote-code trust explicitly in trainer/worker settings. Environment packages are installed on both roles. Workers enforce configured package revisions; coordinator package alignment is operator-owned.

Server `[[sources]]` define tasksets, sampling, groups, retry limits, deterministic finite-taskset shuffling, and train/eval processing. Supported algorithms are GRPO, MaxRL, ECHO, and external-teacher OPD. Supported rollout filters are gibberish, repetition, and zero advantage.
GRPO can use `length_penalty.type = "shortest_correct"` to subtract a bounded flat penalty from correct rollouts whose configured thinking-length metric exceeds the shortest correct rollout in the group. The metric must measure thinking only, and aggregate rewards must be binary.

The coordinator owns trainer output and resume. Distributed trainer configs require LoRA, safetensors, complete checkpoints every step, and no trainer-owned pruning, resume, or partial-load overrides. Server `published_checkpoint_keep_last` can safely bound full-checkpoint retention after durable policy activation; unset retains all checkpoints.

Workers generate and overwrite `<state_dir>/inference.toml`; configure worker-local inference through worker fields rather than maintaining that file.
Use worker `max_model_len` to cap KV-cache context for models whose native context is too large for the rollout GPU.

Server `[policy_distribution]` optionally publishes immutable policies to S3/R2 with `type`, `bucket`, `prefix`, `endpoint_url`, `region`, and `presign_ttl_seconds`. Use the boto3 credential chain, never TOML secrets. Worker policy order is SHARDCAST, approved presigned HTTPS origin, then coordinator fallback. Configure `policy_download_allowed_origins`, `shardcast_servers`, shard concurrency, attempts, fallback, and prefetch interval. Prefetch warms disk only. Worker results default to zstd with two concurrent uploaders; server/source limits are decompressed limits.

The `policy-relay` config requires `coordinator_url`; optional fields are `state_dir`, `port`, polling/request timeouts, `max_versions`, `shard_size_bytes`, and approved policy origins. It requires `AETHER_COORDINATOR_TOKEN` and external TLS.

Use `docs/configuration.md` for field tables and invariants.
