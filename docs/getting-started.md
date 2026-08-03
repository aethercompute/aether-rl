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

Install only the inference worker role on each worker:

```bash
scripts/setup-worker.sh
```

Verifier environments are coordinator-only. The coordinator host also owns any Docker daemon, images, tool dependencies, credentials, and network policy required by their configured harnesses. Workspace environments are opt-in; prefer repeated server package arguments instead of `--all-packages` because research environments can have incompatible dependency pins.

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

## 4. Configure coordinator execution

Each server `[[sources]]` entry identifies and configures its verifier environment. Install that package on the coordinator, set `environment_revision` to the intended package version, and ensure `environment_config` resolves to `environment_id`.

Set server `environment_slots` to the number of episodes the coordinator may execute concurrently. This capacity covers verifier setup, Docker and tool execution, finalization, and scoring; provision coordinator CPU, memory, disk, Docker, and external service limits accordingly. Workers have no environment configuration.

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

Caddy must be installed and operated separately. Ensure the proxy preserves `Authorization`, `Aether-Protocol-Version`, range, content length, and content range headers; permits inference exchange bodies up to the configured limits; and uses timeouts longer than worker long polls. Restrict unauthenticated `/health`, `/ready`, `/docs`, `/redoc`, and `/openapi.json` at the external layer if they must not be public.

A VPN alone does not change the worker's URL rule: a non-loopback `coordinator_url` must still use HTTPS.

## 6. Preflight and launch the coordinator

```bash
scripts/run-server.sh server.toml
```

Server preflight validates configuration, model identity shape, environment configuration plugins and resolved IDs, trainer compatibility, distributed checkpoint requirements, configured policy-store access, and Docker daemon availability when a source uses the Docker runtime. It does not load task data, instantiate environments, execute a sandbox or tool, bind the port, open the production database, load model weights, start the trainer, or test disk capacity.

Wait for liveness and readiness:

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/ready
```

`/health` confirms only that the API process responds. `/ready` also checks the database, active policy integrity, result processing, and trainer health.

Optional S3-compatible policy delivery is configured with server `[policy_distribution]`; use standard boto3 environment credentials and add the generated presigned URL's exact HTTPS origin to worker `policy_download_allowed_origins`. For SHARDCAST, run `scripts/setup-relay.sh` and `scripts/run-relay.sh <relay.toml>`, terminate TLS in front of it, verify `aether-policies.json`, and add its URL to worker `shardcast_servers`. Relay setup is inexact so it preserves taskset packages already selected for the server role in the same checkout. Start the relay after coordinator readiness and before workers. These paths are accelerators; coordinator delivery remains the default fallback.

## 7. Preflight and launch workers

Set `coordinator_url` to the externally reachable HTTPS origin and choose a unique persistent `state_dir` on each worker:

```bash
scripts/run-worker.sh worker.toml https://coordinator.example.com
```

Worker preflight checks GPU visibility and model/tokenizer fingerprints. It does not contact the coordinator, load model weights onto the GPU, start vLLM, make an inference request, or check available disk capacity.

Set worker `inference_slots` to the local vLLM concurrency offered to the coordinator. Set `inference_body_limit_bytes` on both server and workers high enough for the largest inference response; the worker rejects a larger local reply and the coordinator limits inference exchange bodies. Configure the external proxy for the corresponding JSON exchange size.

The worker starts vLLM itself and writes its output to `<state_dir>/inference.log`. It long-polls for inference leases, receives coordinator-generated OpenAI-compatible requests, forwards them to loopback vLLM, and returns replies over outbound HTTPS. Do not launch a separate inference process for a normal worker.

## 8. Check the run

```bash
curl -fsS https://coordinator.example.com/api/v2/status \
  -H "Authorization: Bearer $AETHER_COORDINATOR_TOKEN" \
  -H "Aether-Protocol-Version: 2"
```

Monitor `<run_root>/logs/trainer.log`, coordinator stdout/stderr, worker stdout/stderr, and each worker's `<state_dir>/inference.log`. See [Operations](operations.md) for state layout and restart procedures.
