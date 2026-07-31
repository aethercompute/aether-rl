# Getting started

This guide launches a configured run with one coordinator and one or more workers. Run configurations are workload-specific and are not checked in; prepare `server.toml`, `worker.toml`, and `trainer.toml` using the [configuration reference](configuration.md).

## 1. Prepare every machine

Install Linux, a compatible NVIDIA driver/CUDA stack, Python 3.12, Git, and `uv >= 0.11.1`. Clone the top-level repository; the role setup scripts initialize recursive submodules over HTTPS:

```bash
git clone https://github.com/aethercompute/aether-rl.git
cd aether-rl
```

Use persistent local paths for the server `run_root` and each worker `state_dir`. Do not reuse one worker state directory across machines or concurrent processes.

## 2. Install the coordinator

The coordinator loads source tasksets, so install every environment package referenced by `server.toml`:

```bash
export ENVIRONMENT_PACKAGE='your-environment-package'
scripts/setup-server.sh "$ENVIRONMENT_PACKAGE"
```

Install the worker role and every advertised environment package on each worker:

```bash
scripts/setup-worker.sh "$ENVIRONMENT_PACKAGE"
```

Workspace environments are opt-in. Prefer repeated `--package <environment>` arguments instead of `--all-packages`; research environments can have incompatible dependency pins.

## 3. Pin model identity

Resolve the model to a full lowercase 40-character Hugging Face commit. Branches, tags, and mutable local paths are not valid run identities. Generate canonical fingerprints with the same implementation workers use:

```bash
export MODEL_REPOSITORY='organization/model'
export MODEL_REVISION='<40-character-commit>'
uv run model-identity \
  --model-name "$MODEL_REPOSITORY" \
  --model-revision "$MODEL_REVISION"
```

If the tokenizer is in another repository or commit, add `--tokenizer-name` and `--tokenizer-revision`, then set the same values explicitly in the trainer's `[tokenizer]` table. Use `--trust-remote-code` only after reviewing the repository; also set `trust_remote_code = true` in worker config and trainer `[model]` so runtime loading matches identity generation. Private repositories use the normal Hugging Face authentication environment.

Copy the generated `[base_model]` table into both server and worker configurations. Set `[model].name` and `[model].revision` in `trainer.toml` to the same model identity. All workers in a run must use identical values.

Only unquantized base models with `quantization = "none"` are currently accepted by worker identity verification.

## 4. Configure the environment

The coordinator source and worker environment must agree on environment ID, revision, and resolved verifier configuration. The configured package revision must equal the installed package version.

Keep each configured environment package and version installed on both roles. Workers enforce the installed package version; the coordinator resolves and loads its configured taskset but cannot verify that the advertised revision matches a package version. Update both `[[sources]]` in `server.toml` and `[[environments]]` in `worker.toml` together.

## 5. Secure the coordinator

Generate one high-entropy ASCII bearer token and provide it to the coordinator and every worker through the environment:

```bash
export AETHER_COORDINATOR_TOKEN='<random-secret>'
```

For a local smoke run, loopback HTTP is allowed. For remote workers, keep the coordinator bound to loopback and terminate HTTPS externally. A minimal Caddy site is:

```caddyfile
coordinator.example.com {
    reverse_proxy 127.0.0.1:8080
}
```

Caddy must be installed and operated separately. Ensure the proxy preserves `Authorization` and `Aether-Protocol-Version`, permits result bodies up to `result_body_limit_bytes`, and uses timeouts longer than worker long polls. Restrict unauthenticated `/health`, `/ready`, `/docs`, `/redoc`, and `/openapi.json` at the external layer if they must not be public.

A VPN alone does not change the worker's URL rule: a non-loopback `coordinator_url` must still use HTTPS.

## 6. Preflight and launch the coordinator

```bash
scripts/run-server.sh server.toml
```

Server preflight validates configuration, identity shape, source resolution, trainer compatibility, and distributed checkpoint requirements. It does not bind the port, open the production database, load model weights, start the trainer, or test disk capacity.

Wait for liveness and readiness:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/ready
```

`/health` confirms only that the API process responds. `/ready` also checks the database, active policy integrity, result processing, and trainer health.

Optional S3-compatible policy delivery is configured with server `[policy_distribution]`; use standard boto3 environment credentials and add the generated presigned URL's exact HTTPS origin to worker `policy_download_allowed_origins`. For SHARDCAST, run `scripts/setup-relay.sh` and `scripts/run-relay.sh <relay.toml>`, terminate TLS in front of it, verify `aether-policies.json`, and add its URL to worker `shardcast_servers`. Relay setup is inexact so it preserves taskset packages already selected for another role in the same checkout. Start the relay after coordinator readiness and before workers. These paths are accelerators; coordinator delivery remains the default fallback.

## 7. Preflight and launch workers

Set `coordinator_url` to the externally reachable HTTPS origin and choose a unique persistent `state_dir` on each worker:

```bash
scripts/run-worker.sh worker.toml https://coordinator.example.com
```

Worker preflight checks GPU visibility, model/tokenizer fingerprints, installed environment versions, and environment resolution. It does not contact the coordinator, load model weights onto the GPU, start vLLM, execute an episode, or check available disk capacity.

Proxies must preserve authorization, protocol, content encoding, range, content length, and content range headers. Configure body limits for decompressed results when zstd uploads are enabled.

The worker starts vLLM itself and writes its output to `<state_dir>/inference.log`. Do not launch a separate inference process for a normal worker.

## 8. Check the run

```bash
curl -fsS https://coordinator.example.com/api/v1/status \
  -H "Authorization: Bearer $AETHER_COORDINATOR_TOKEN" \
  -H "Aether-Protocol-Version: 1"
```

Monitor `<run_root>/logs/trainer.log`, coordinator stdout/stderr, worker stdout/stderr, and each worker's `<state_dir>/inference.log`. See [Operations](operations.md) for state layout and restart procedures.
