#!/usr/bin/env bash
# Run one model (or set) on one dataset in the foreground, PIDSMaker style.
#   scripts/run_local.sh MODEL DATASET [--n-jobs N] [--task T] [--stage.key=value ...]
# Examples
#   scripts/run_local.sh A+R paper19
#   scripts/run_local.sh core GS027 --fit.n_starts=40 --n-jobs 8
#   scripts/run_local.sh S+C,A+C paper19 --preprocess.correction=include
#   scripts/run_local.sh A+R GS040 --task ppc --ppc.n_sims=100
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -lt 2 ]]; then sed -n '2,8p' "$0"; exit 1; fi
source "$HERE/scripts/activate.sh" >/dev/null
cd "$HERE"
exec python -m rl_ladder run "$@"
