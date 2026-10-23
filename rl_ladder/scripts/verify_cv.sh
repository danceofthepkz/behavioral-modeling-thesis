#!/usr/bin/env bash
# Verify saved CV parameters and scores; never invokes a fitting job.
#   scripts/verify_cv.sh RUN_NAME_OR_PLAN_JSON
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/scripts/activate.sh" >/dev/null
cd "$HERE"
exec python -m rl_ladder verify-cv "${1:?pass a run name or plan.json path}"
