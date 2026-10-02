#!/usr/bin/env bash
set -euo pipefail
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$BASE_DIR/../.." && pwd)"
PY="$REPO_ROOT/.venv-phase3/bin/python"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
[[ -x "$PY" ]] || { echo "ERROR: missing $PY" >&2; exit 2; }
exec "$PY" "$BASE_DIR/collect_demos.py" "$@"
