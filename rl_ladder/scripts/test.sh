#!/usr/bin/env bash
# Run the test suite. Data-dependent replication tests run when RL_LADDER_DATA is set.
#   scripts/test.sh [pytest options]
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/scripts/activate.sh" >/dev/null
cd "$HERE"
exec python -m pytest -q tests "$@"
