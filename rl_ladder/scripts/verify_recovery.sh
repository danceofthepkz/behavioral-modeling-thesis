#!/usr/bin/env bash
# Replay saved synthetic trials and selected scores; no fitting or Slurm jobs.
# Usage: bash scripts/verify_recovery.sh RUN_NAME_OR_PLAN_JSON
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
source "$HERE/scripts/activate.sh" >/dev/null
cd "$HERE"
exec python -m rl_ladder verify-recovery "${1:?pass a recovery run name or plan.json path}"
