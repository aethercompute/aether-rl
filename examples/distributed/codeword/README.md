# Distributed codeword learning proof

This experiment is a four-association RL pipeline smoke test for the complete Aether RL path, using a 3090 coordinator/trainer and a remote 5090 rollout worker. `Qwen/Qwen2.5-0.5B-Instruct` must learn a fixed hidden mapping from four marker letters to four codewords:

```text
A -> dax
B -> wug
C -> zorp
D -> kiv
```

The model sees the marker and the four allowed codewords, but not the mapping. Training and evaluation use different prompt IDs, while sharing the four semantic marker classes the model must learn. Both splits are balanced. Reward accepts a response containing exactly one codeword, while `exact_format` requires the entire response to be that codeword. Evaluation never enters trainer batches. This proves remote RL optimization and learned conditional behavior, not broad task generalization.

## Expected cost

The default run performs 10 LoRA optimizer steps. An update requires 128 surviving train rollouts, or eight complete 16-sample GRPO groups. The 4:1 train/eval scheduling weights target approximately 32 eval rollouts per policy; completed evals can arrive after the next policy activates while retaining their original policy identity. Completions are capped at 8 tokens. Full resumable checkpoints are retained every step, while optional full-weight exports are disabled.

## 1. Prepare the server

```bash
uv sync --group server --package aether-rl --package codeword-v1

uv run model-identity \
  --model-name Qwen/Qwen2.5-0.5B-Instruct \
  --model-revision 7ae557604adf67be50417f59c2c2f167def9a775
```

The checked-in configs are pinned to `7ae557604adf67be50417f59c2c2f167def9a775`. Regenerate and replace both `[base_model]` blocks if you choose another commit, then put the same commit in trainer `[model].revision`.

Set `WANDB_API_KEY` before server launch to enable trainer logging:

```bash
export WANDB_API_KEY='<your-key>'
```

The 3090 machine uses:

```bash
export AETHER_COORDINATOR_TOKEN='<random-ascii-secret>'
scripts/preflight-server.sh @ examples/distributed/codeword/server.toml
scripts/launch-server.sh @ examples/distributed/codeword/server.toml
```

Keep the coordinator bound to loopback and expose it through an external HTTPS proxy or VPN gateway. Wait for `/ready` before starting workers.

## 2. Prepare the rollout worker

On the 5090 machine:

```bash
uv sync --group worker --package aether-rl --package codeword-v1
```

Copy the completed `worker.toml`, set `coordinator_url` to the server's HTTPS origin, and choose a persistent local `state_dir`. Then run:

```bash
export AETHER_COORDINATOR_TOKEN='<same-random-ascii-secret>'
scripts/preflight-worker.sh @ examples/distributed/codeword/worker.toml
scripts/launch-worker.sh @ examples/distributed/codeword/worker.toml
```

Each worker uses four rollout slots on one GPU. Two workers provide eight aggregate slots. Give every worker its own persistent local `state_dir`; the same path is fine on separate machines.

## 3. Observe learning

While the run is active, summarize immutable held-out eval traces on the server:

```bash
uv run eval-report \
  --run-root outputs/codeword-qwen25-stable-proof/server \
  --source-id codeword-eval
```

For this binary reward, `mean_reward` is accuracy including failed rollouts. `effective_mean_reward` excludes errored rollouts, and `exact_format_mean` measures compliance with the one-codeword response format.

Policy 0 is the unmodified base model. The predefined comparison target is policy 9. Policy 10 is terminal and normally has no eval groups because reaching `max_steps` gates new leases.

The proof passes when:

- Policy 0 has at least 12 eval rollouts.
- Policy 9 has at least 24 eval rollouts.
- Policy 0 `mean_reward` is below `0.50`.
- Policy 9 `mean_reward` is at least `0.80`.
- Improvement is at least `0.40`.
- Policy 9 `exact_format_mean` is at least `0.95`.
- Eval errors remain zero or are explained operationally.
- Policies, checkpoints, and trainer metrics advance with optimizer steps.

Supporting artifacts are:

```text
outputs/codeword-qwen25-stable-proof/server/
├── coordinator.sqlite
├── logs/trainer.log
├── policies/
├── training-queue/groups/
└── trainer/
    ├── metrics.jsonl
    └── checkpoints/
```

## 4. Test dynamic workers

After several policy versions have been published:

1. Hard-stop the 5090 worker while assignments are active to test lease expiry; a graceful stop may cancel work before producing a result.
2. Confirm `/api/v1/status` shows stale sessions or active leases until the lease timeout.
3. Wait for expired work to be requeued.
4. Restart the worker with the same `state_dir` and confirm policy/eval progress resumes.
5. Start another worker with a different persistent `state_dir`; on the same host, also use a different `inference_port`.
6. Stop either worker and confirm the other continues leasing work.

Lease reassignment does not guarantee a pending spool entry: active execution may have stopped before producing one. To test spool recovery separately, block coordinator connectivity, wait until `<state_dir>/spool/pending/` contains a completed result, hard-stop and restart the worker, restore connectivity, and confirm that entry drains exactly once. Late results from expired leases may move to `spool/rejected/`; this is expected fail-closed behavior.

## Interpreting failure

- No improvement with near-zero entropy usually means the initial policy did not explore multiple codewords. Increase train temperature or group size and start a new run root.
- High effective accuracy with many errors is not a pass; use all-attempt `mean_reward` as the primary result.
- Improving training loss without held-out reward improvement does not prove learning.
- Do not tune against the eval split after inspecting individual answers; create a new split or mapping for another confirmatory run.

Run live eval reporting in a separate server terminal. It creates a small W&B run with reward, exact-format rate, rollout count, and errors plotted against policy version:

```bash
uv run eval-report \
  --run-root outputs/codeword-qwen25-stable-proof/server \
  --source-id codeword-eval \
  --wandb-project aether-rl-codeword-stable \
  --wandb-name qwen2.5-0.5b-codeword-stable-eval \
  --wandb-group distributed-learning-proof \
  --watch-seconds 20
```

Stop the reporter with `Ctrl+C` after training completes. The trainer W&B run uses an explicit nine-metric allowlist and does not create the broad automatic overview workspace.
