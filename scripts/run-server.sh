#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  printf 'usage: %s path/to/server.toml [--overrides...]\n' "$0" >&2
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
  printf 'server config does not exist: %s\n' "$CONFIG" >&2
  exit 2
fi
CONFIG=$(realpath "$CONFIG")

cd "$ROOT"
printf 'Preflighting server config: %s\n' "$CONFIG"
"$ROOT/scripts/preflight-server.sh" @ "$CONFIG" "$@"
printf 'Starting server.\n'
exec "$ROOT/scripts/launch-server.sh" @ "$CONFIG" "$@"
