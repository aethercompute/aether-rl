# 50-step distributed reverse-text run

This run extends the Qwen2.5-0.5B reverse-text learning proof to 50 optimizer steps. Each step consumes 64 train rollouts in eight-sample GRPO groups. Rollout workers execute up to 32 assignments concurrently, and train work is scheduled at a 16:1 ratio over online evaluation work.

Every step retains a complete checkpoint and immutable LoRA policy. Start with at least 150 GiB of free server storage; 200 GiB is safer. Use a new run root, and do not delete or alter another run's state while its coordinator is running.

## Server

Stop the previous coordinator cleanly before reusing port 8080. Set secrets without putting their values in the config files or command history:

```bash
scripts/setup-server.sh reverse-text-proof-v1
read -s AETHER_COORDINATOR_TOKEN
export AETHER_COORDINATOR_TOKEN
read -s WANDB_API_KEY
export WANDB_API_KEY
df -h /
scripts/run-server.sh examples/distributed/reverse-text-50step/server.toml
```

Keep the coordinator bound to loopback and expose it through a stable HTTPS reverse proxy. The worker URL below uses `https://coordinator.example.com` as a placeholder.

## W&B evaluation

The trainer automatically publishes optimization metrics to the `aether-rl-reverse-text` W&B project. In a separate server terminal, publish durable evaluation metrics by policy version:

```bash
read -s WANDB_API_KEY
export WANDB_API_KEY
uv run --no-default-groups eval-report \
  --run-root outputs/reverse-text-qwen25-50step/server \
  --source-id reverse-text-50step-eval \
  --wandb-project aether-rl-reverse-text \
  --wandb-name qwen2.5-0.5b-reverse-text-50step-eval \
  --wandb-group distributed-learning-proof \
  --watch-seconds 20
```

## Workers

Run these commands on each rollout machine after the coordinator is reachable. A 5090 worker must report 32 GiB and use a CUDA 12.8-or-newer PyTorch build.

```bash
scripts/setup-worker.sh reverse-text-proof-v1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
uv run --no-default-groups --group worker python -c \
  'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
read -s AETHER_COORDINATOR_TOKEN
export AETHER_COORDINATOR_TOKEN
scripts/run-worker.sh \
  examples/distributed/reverse-text-50step/worker.toml \
  https://coordinator.example.com
```

Use the same command on every separate worker machine. Each machine has its own local state directory and stable worker identity. Do not launch multiple worker processes against the same `state_dir`.

## Monitor

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/ready
curl -fsS http://127.0.0.1:8080/api/v1/status \
  -H "Authorization: Bearer $AETHER_COORDINATOR_TOKEN" \
  -H "Aether-Protocol-Version: 1"
```

On the server, monitor `outputs/reverse-text-qwen25-50step/server/logs/trainer.log` and `outputs/reverse-text-qwen25-50step/server/trainer/metrics.jsonl`. On each worker, monitor `outputs/reverse-text-qwen25-50step/worker/inference.log`.

After the first two post-warmup steps, check `time/wait_for_batch` in the trainer metrics. Stop and diagnose the rollout path if it remains above 30 seconds or the worker GPU stays mostly idle.
