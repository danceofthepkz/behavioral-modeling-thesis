#!/usr/bin/env bash
# Run an experiment file (config/experiments/NAME.yml) on this machine.
#   scripts/run_experiment.sh NAME [--n-jobs N] [--dry-run] [--stage.key=value ...]
# Examples
#   scripts/run_experiment.sh replicate_paper --n-jobs 8
#   scripts/run_experiment.sh core_ladder --n-jobs 8
#   scripts/run_experiment.sh param_recovery --n-jobs 8 --validate.n_reps=20
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -lt 1 ]]; then sed -n '2,7p' "$0"; ls "$HERE/config/experiments"; exit 1; fi
source "$HERE/scripts/activate.sh" >/dev/null
cd "$HERE"
exec python -m rl_ladder experiment "$@"
