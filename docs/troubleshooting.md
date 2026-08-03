# Troubleshooting

Start with coordinator `/health`, `/ready`, authenticated `/api/v2/status` using protocol version 2, `<run_root>/logs/trainer.log`, worker process output, and `<state_dir>/inference.log`.

## Server preflight fails

### Placeholder or mismatched model identity

Generate a fresh block from full immutable commits:

```bash
uv run model-identity --model-name <repo> --model-revision <40-character-commit>
```

Use the same block on server and workers and the same model name/revision in trainer config. If tokenizer name or revision differs from the model, set the matching values explicitly under trainer `[tokenizer]`. Mirror any `--trust-remote-code` choice in worker config and trainer `[model]`. Ensure Hugging Face credentials and cache access are available.

### Environment identity mismatch

Install the environment package on the coordinator, manually align its version with `environment_revision`, and ensure `environment_config` resolves to `environment_id`. Coordinator preflight imports the environment configuration plugin and checks Docker daemon availability for Docker-backed sources, but does not load task data or compare the installed package version. Infinite tasksets also require `task_limit`.

### Trainer rejected by distributed validation

The trainer must use LoRA, safetensors publication, and a complete unpruned checkpoint every step. Remove configured resume steps, checkpoint output overrides, pruning, `weights_only`, and all checkpoint skip flags.

## Worker preflight fails

### No GPU or tensor-parallel mismatch

Verify `CUDA_VISIBLE_DEVICES`, NVIDIA driver health, and that `tensor_parallel_size` does not exceed visible GPUs. Preflight does not prove model weights fit in memory.

### Fingerprint mismatch

Regenerate identity on the installed Aether RL revision. Tokenizer library changes can alter canonical fingerprints, so use the same project revision across the fleet.

## Worker cannot connect

- Remote `coordinator_url` must be an HTTPS origin with no path or query.
- Confirm DNS, certificate trust, firewall egress, and proxy body/time limits.
- Confirm both roles use the same ASCII `AETHER_COORDINATOR_TOKEN`.
- Ensure the proxy preserves `Authorization` and `Aether-Protocol-Version: 2`.
- `401` indicates a missing or incorrect token; `400` can indicate a protocol-version problem.
- `503 trainer_unavailable` means readiness, result processing, trainer health, or active-policy integrity has failed.

Worker preflight intentionally does not test coordinator connectivity.

## `/health` passes but `/ready` fails

`/health` only means the API responds. Check authenticated status and trainer log. Common causes are trainer exit, failed result processing, database verification failure, or corrupted active-policy files. New leases remain gated until readiness is restored.

## Worker-local vLLM does not start

Inspect `<state_dir>/inference.log`. Check GPU memory, model access, CUDA compatibility, `tensor_parallel_size`, LoRA rank limits, and whether another process owns `inference_port`. The worker does not automatically restart vLLM after an unexpected exit; restart the worker after fixing the cause.

`cudaErrorUnsupportedPtxVersion` means a bundled CUDA kernel was compiled by a
toolchain newer than the host NVIDIA driver supports. The pinned x86_64 vLLM wheel
uses CUDA 12.9, which requires Linux driver 575.51.03 or newer. Upgrade the host
driver or use a machine image with a compatible driver, then restart the worker
with its existing state directory. Reinstalling the CUDA toolkit inside a
container does not change the host driver's PTX support.

## Reward and gradients remain zero

Inspect raw rollout responses before continuing. Trainer metrics with
`optim/zero_grad_ratio = 1`, `optim/grad_norm = 0`, and zero loss mean every group
has uniform advantages and published policies are no-ops. Raw `Ġ` and `Ċ`
characters in responses indicate that a byte-level tokenizer was loaded through
the wrong decoder. Otherwise compare mathematically correct responses against the
environment's answer-extraction and format contracts. After correcting the cause,
start a fresh run identity instead of resuming checkpoints created from zero
gradients.

## Episodes are not progressing

Inspect status counts for active inference leases, accepted/pending results, processing results, and groups. Confirm workers remain connected, inspect worker inference logs, and check coordinator capacity and permissions for SQLite, result artifacts, training queue, policies, and trainer output.

If environments wait on model calls, compare server `environment_slots` with total available worker `inference_slots`. Check external proxy timeouts and ensure server and worker `inference_body_limit_bytes` both accommodate the largest response. The worker exchanges inference requests and replies as identity-encoded JSON over `/api/v2/inference/exchange`.

Verifier environments, Docker sandboxes, tools, episode finalization, and scoring all run on the coordinator. Diagnose those failures in coordinator output and provision Docker images, credentials, network access, CPU, memory, and disk there, not on workers. Completed episodes become durable only in coordinator state; preserve the complete `run_root`.

## External policy download fails

An empty or mismatched `policy_download_allowed_origins` disables presigned URLs. Match the exact HTTPS scheme and authority from the generated URL; redirects are intentionally refused. Check coordinator boto3 credentials, bucket permissions, endpoint and region, clock skew, presigned expiry, and object SHA-256 metadata. Server preflight probes bucket access. Workers retry external downloads with fresh locations and use coordinator delivery on the final configured attempt when fallback is enabled.

For SHARDCAST, verify the relay token, relay process output, HTTPS proxy, and `aether-policies.json`. A stale index, evicted version, missing shard, or digest mismatch causes the worker to try the next configured transport. Prefetch only warms the verified disk cache and does not load the adapter into vLLM.

## Restart does not resume

Confirm the complete run root and trainer output are present and that the active policy version has its full checkpoint. Do not point a new run at an old database, change source definitions under existing IDs, or set trainer `ckpt.resume_step` manually. Only one coordinator may hold a run root.

If the trainer repeatedly reports a missing `run_<id>/control/orch.toml`, compare the logged path with the directory exported under the trainer output. Run directory names preserve the complete `run_id`, including dots; a shortened name indicates mismatched or outdated coordinator/trainer code.

For workers, preserve the original `state_dir` for stable identity and cached adapters. Starting two workers against one state directory fails its process lock and risks operational confusion. Worker restart does not resume active inference leases; coordinator-owned episode attempts fail or expire and are retried centrally. Coordinator restart retains accepted results but reruns episodes that had not reached durable acceptance.

## Disk growth

Trainer-owned checkpoint pruning is unsafe because publication may not have consumed the next stable version. Server `published_checkpoint_keep_last` enables coordinator-owned pruning after durable policy activation; unset retains every checkpoint. Plan capacity for retained full checkpoints, immutable policy adapters, accepted traces, and worker caches. Do not manually delete active or unpublished checkpoints or referenced policy files.
