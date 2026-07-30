#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'usage: %s path/to/policy-relay.toml [--overrides...]\n' "$0" >&2
  exit 2
fi
if [ -z "${AETHER_COORDINATOR_TOKEN:-}" ]; then
  printf 'AETHER_COORDINATOR_TOKEN is required\n' >&2
  exit 2
fi

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG=$1
shift
if [ ! -f "$CONFIG" ]; then
  printf 'policy relay config does not exist: %s\n' "$CONFIG" >&2
  exit 2
fi
CONFIG=$(realpath "$CONFIG")

cd "$ROOT"
printf 'Starting policy relay: %s\n' "$CONFIG"
exec "$ROOT/scripts/launch-relay.sh" @ "$CONFIG" "$@"
