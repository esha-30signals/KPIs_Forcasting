#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DAGSTER_HOME="${SCRIPT_DIR}/.dagster_home_local"
mkdir -p "$DAGSTER_HOME"
exec dagster dev -m orchestration.definitions "$@"
