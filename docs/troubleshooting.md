# Troubleshooting

Start with coordinator `/health`, `/ready`, authenticated `/api/v1/status`, `<run_root>/logs/trainer.log`, worker process output, and `<state_dir>/inference.log`.

## Server preflight fails

### Placeholder or mismatched model identity

Generate a fresh block from full immutable commits:

```bash
uv run model-identity --model-name <repo> --model-revision <40-character-commit>
```

Use the same block on server and workers and the same model name/revision in trainer config. If tokenizer name or revision differs from the model, set the matching values explicitly under trainer `[tokenizer]`. Mirror any `--trust-remote-code` choice in worker config and trainer `[model]`. Ensure Hugging Face credentials and cache access are available.

### Environment identity mismatch

Install the environment package on the coordinator, manually align its version with `environment_revision`, and ensure `environment_config` resolves to `environment_id`. Coordinator preflight resolves the taskset but does not compare its installed package version. Infinite tasksets also require `task_limit`.

### Trainer rejected by distributed validation

The trainer must use LoRA, safetensors publication, and a complete unpruned checkpoint every step. Remove configured resume steps, checkpoint output overrides, pruning, `weights_only`, and all checkpoint skip flags.

## Worker preflight fails

### No GPU or tensor-parallel mismatch

Verify `CUDA_VISIBLE_DEVICES`, NVIDIA driver health, and that `tensor_parallel_size` does not exceed visible GPUs. Preflight does not prove model weights fit in memory.

### Package revision mismatch

Install every package named in `[[environments]]` and set `revision` to its exact installed package version. Keep worker environment entries sorted and unique by ID and revision.

### Fingerprint mismatch

Regenerate identity on the installed Aether RL revision. Tokenizer library changes can alter canonical fingerprints, so use the same project revision across the fleet.

## Worker cannot connect

- Remote `coordinator_url` must be an HTTPS origin with no path or query.
- Confirm DNS, certificate trust, firewall egress, and proxy body/time limits.
- Confirm both roles use the same ASCII `AETHER_COORDINATOR_TOKEN`.
- Ensure the proxy preserves `Authorization` and `Aether-Protocol-Version`.
- `401` indicates a missing or incorrect token; `400` can indicate a protocol-version problem.
- `503 trainer_unavailable` means readiness, result processing, trainer health, or active-policy integrity has failed.

Worker preflight intentionally does not test coordinator connectivity.

## `/health` passes but `/ready` fails

`/health` only means the API responds. Check authenticated status and trainer log. Common causes are trainer exit, failed result processing, database verification failure, or corrupted active-policy files. New leases remain gated until readiness is restored.

## Worker-local vLLM does not start

Inspect `<state_dir>/inference.log`. Check GPU memory, model access, CUDA compatibility, `tensor_parallel_size`, LoRA rank limits, and whether another process owns `inference_port`. The worker does not automatically restart vLLM after an unexpected exit; restart the worker after fixing the cause.

## Results are not progressing

Inspect status counts for active leases, accepted/pending results, processing results, and groups. Confirm workers remain connected and their local pending spool is below `spool_max_entries`. Check server disk capacity and permissions for SQLite, spools, training queue, policies, and trainer output.

Do not delete pending worker spool files. They are the durable copy until coordinator acknowledgment.

For zstd uploads, ensure the proxy preserves `Content-Encoding: zstd` rather than decompressing the body while forwarding the header. Result limits apply to decompressed bytes. If increasing `result_upload_concurrency`, also check coordinator/proxy connection limits and worker spool pressure.

## External policy download fails

An empty or mismatched `policy_download_allowed_origins` disables presigned URLs. Match the exact HTTPS scheme and authority from the generated URL; redirects are intentionally refused. Check coordinator boto3 credentials, bucket permissions, endpoint and region, clock skew, presigned expiry, and object SHA-256 metadata. Server preflight probes bucket access. Workers retry external downloads with fresh locations and use coordinator delivery on the final configured attempt when fallback is enabled.

For SHARDCAST, verify the relay token, relay process output, HTTPS proxy, and `aether-policies.json`. A stale index, evicted version, missing shard, or digest mismatch causes the worker to try the next configured transport. Prefetch only warms the verified disk cache and does not load the adapter into vLLM.

## Rejected worker spool entries

Files under `<state_dir>/spool/rejected/` represent nonretryable submissions and are not automatically resent. Preserve them for diagnosis. Typical causes include expired leases, identity conflicts, malformed traces, body limits, or incompatible policy reporting.

## Restart does not resume

Confirm the complete run root and trainer output are present and that the active policy version has its full checkpoint. Do not point a new run at an old database, change source definitions under existing IDs, or set trainer `ckpt.resume_step` manually. Only one coordinator may hold a run root.

If the trainer repeatedly reports a missing `run_<id>/control/orch.toml`, compare the logged path with the directory exported under the trainer output. Run directory names preserve the complete `run_id`, including dots; a shortened name indicates mismatched or outdated coordinator/trainer code.

For workers, preserve the original `state_dir`. Starting two workers against one state directory fails its process lock and risks operational confusion.

## Disk growth

Trainer-owned checkpoint pruning is unsafe because publication may not have consumed the next stable version. Server `published_checkpoint_keep_last` enables coordinator-owned pruning after durable policy activation; unset retains every checkpoint. Plan capacity for retained full checkpoints, immutable policy adapters, accepted traces, and worker caches. Do not manually delete active or unpublished checkpoints or referenced policy files.
