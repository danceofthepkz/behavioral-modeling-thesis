#!/usr/bin/env bash
# Submit one stage of the primary 18-model protocol. Each stage is explicit.
# Usage: scripts/slurm/submit_main_ladder.sh {smoke|full|new|model-recovery|param-recovery} [submit.sh options]
set -eo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
mode="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    smoke)
        exp=smoke_main_ladder; chunk=4; maxrun=2; walltime=00:30:00 ;;
    full)
        exp=main_ladder; chunk=4; maxrun=8; walltime=02:00:00 ;;
    new)
        exp=main_new_mice; chunk=4; maxrun=8; walltime=02:00:00 ;;
    model-recovery)
        exp=main_model_recovery; chunk=2; maxrun=8; walltime=02:00:00 ;;
    param-recovery)
        exp=main_param_recovery; chunk=4; maxrun=8; walltime=02:00:00 ;;
    *)
        echo "Usage: scripts/slurm/submit_main_ladder.sh {smoke|full|new|model-recovery|param-recovery} [submit.sh options]"
        exit 1 ;;
esac
exec bash "$HERE/scripts/slurm/submit.sh" "$exp" --name-suffix _v023 \
    --chunk "$chunk" --max-running "$maxrun" --time "$walltime" --mem 2G "$@"
