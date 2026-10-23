#!/usr/bin/env bash
# Submit CV-only provenance runs. This command does not submit full-data fits.
#   scripts/slurm/submit_cv_audit.sh smoke [submit.sh options]
#   scripts/slurm/submit_cv_audit.sh full  [submit.sh options]
set -eo pipefail
HERE="$(cd "$(dirname "$0")/../.." && pwd)"
mode="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$mode" in
    smoke)
        exec bash "$HERE/scripts/slurm/submit.sh" smoke_cv_audit --name-suffix _v021 \
            --chunk 3 --max-running 2 --time 00:15:00 --mem 2G "$@"
        ;;
    full)
        exec bash "$HERE/scripts/slurm/submit.sh" feedback_cv_audit --name-suffix _v021 \
            --chunk 5 --max-running 8 --time 01:00:00 --mem 2G "$@"
        ;;
    *)
        echo "Usage: scripts/slurm/submit_cv_audit.sh {smoke|full} [submit.sh options]"
        exit 1
        ;;
esac
