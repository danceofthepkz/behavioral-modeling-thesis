#!/usr/bin/env bash
# Complete S-family R/C/F fitting and chronological CV, in new run folders.
#   scripts/slurm/submit_reward_ladder.sh smoke [submit.sh options]
#   scripts/slurm/submit_reward_ladder.sh full  [submit.sh options]
set -eo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
mode="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    smoke)
        exec bash "$HERE/scripts/slurm/submit.sh" smoke_reward_ladder --name-suffix _v022 \
            --chunk 4 --max-running 2 --time 00:30:00 --mem 2G "$@"
        ;;
    full)
        exec bash "$HERE/scripts/slurm/submit.sh" reward_ladder --name-suffix _v022 \
            --chunk 4 --max-running 8 --time 02:00:00 --mem 2G "$@"
        ;;
    *)
        echo "Usage: scripts/slurm/submit_reward_ladder.sh {smoke|full} [submit.sh options]"
        exit 1
        ;;
esac
