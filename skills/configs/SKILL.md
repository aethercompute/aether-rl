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

Canonical templates are `examples/distributed/reverse-text/server.toml`, `worker.toml`, and `trainer.toml`. Their all-zero identities intentionally fail preflight. Generate the shared identity block with:

```bash
uv run model-identity --model-name <repo> --model-revision <40-character-commit>
```

The server and all workers require the same full model/tokenizer revisions, fingerprints, vocabulary size, and quantization identity. The trainer model name/revision must match; configure a distinct tokenizer and remote-code trust explicitly in trainer/worker settings. Environment packages are installed on both roles. Workers enforce configured package revisions; coordinator package alignment is operator-owned.

Server `[[sources]]` define tasksets, sampling, groups, retry limits, and train/eval processing. Supported algorithms are GRPO, MaxRL, ECHO, and external-teacher OPD. Supported rollout filters are gibberish, repetition, and zero advantage.

The coordinator owns trainer output and resume. Distributed trainer configs require LoRA, safetensors, complete unpruned checkpoints every step, and no resume or partial-load overrides.

Workers generate and overwrite `<state_dir>/inference.toml`; configure worker-local inference through worker fields rather than maintaining that file.

Use `docs/configuration.md` for field tables and invariants.
