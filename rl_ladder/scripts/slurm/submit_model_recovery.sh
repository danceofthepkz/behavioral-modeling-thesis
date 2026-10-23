#!/usr/bin/env bash
# Explicit stages; submitting smoke never submits a full run automatically.
# Usage: bash scripts/slurm/submit_model_recovery.sh {smoke|old|new} [submit.sh options]
set -eo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
mode="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    smoke) exp=smoke_model_recovery; chunk=1; maxrun=4; walltime=00:30:00 ;;
    old) exp=recovery_main_old; chunk=3; maxrun=8; walltime=04:00:00 ;;
    new) exp=recovery_main_new; chunk=2; maxrun=8; walltime=04:00:00 ;;
    *) echo "Usage: $0 {smoke|old|new} [submit.sh options]"; exit 1 ;;
esac
exec bash "$HERE/scripts/slurm/submit.sh" "$exp" --name-suffix _v024 \
    --chunk "$chunk" --max-running "$maxrun" --time "$walltime" --mem 2G "$@"
