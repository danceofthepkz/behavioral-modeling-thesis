#!/usr/bin/env bash
# Same as run_local.sh, but in the background (nohup), logging to logs/.
#   scripts/run.sh MODEL DATASET [options] [--stage.key=value ...]
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -lt 2 ]]; then sed -n '2,3p' "$0"; exit 1; fi
source "$HERE/scripts/activate.sh" >/dev/null
cd "$HERE"
mkdir -p logs
log="logs/$(date +%Y%m%d_%H%M%S)_$(printf '%s_%s' "$1" "$2" | tr '+,/' '__-').log"
nohup python -m rl_ladder run "$@" >"$log" 2>&1 &
echo "started PID $!  log: $log   (follow with: tail -f $log)"
