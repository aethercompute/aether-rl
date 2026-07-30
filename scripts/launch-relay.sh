#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'usage: %s @ path/to/policy-relay.toml [--overrides...]\n' "$0" >&2
  exit 2
fi
if [ -z "${AETHER_COORDINATOR_TOKEN:-}" ]; then
  printf 'AETHER_COORDINATOR_TOKEN is required\n' >&2
  exit 2
fi

exec uv run --no-default-groups --group relay policy-relay "$@"
