#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

git -c url.https://github.com/.insteadOf=git@github.com: \
  -c url.https://github.com/.insteadOf=ssh://git@github.com/ \
  submodule update --init --recursive

packages=(--package aether-rl)
for package in "$@"; do
  packages+=(--package "$package")
done

uv sync --no-default-groups --group worker "${packages[@]}"
printf 'Worker setup complete.\n'
