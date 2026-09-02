#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UV_BIN="$(command -v uv || true)"

if [[ -z "$UV_BIN" ]]; then
  echo "Error: uv was not found in PATH." >&2
  exit 127
fi

cd "$ROOT_DIR"
exec "$UV_BIN" run python server.py "$@"
