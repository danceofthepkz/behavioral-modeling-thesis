#!/usr/bin/env bash
# Package the code (no data, no results) for upload to Firebird.
#   scripts/pack_for_firebird.sh            -> dist/rl_ladder_<stamp>.tar.gz
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/.."
mkdir -p "$HERE/dist"
out="$HERE/dist/rl_ladder_$(date +%Y%m%d_%H%M).tar.gz"
COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata -czf "$out" \
    --exclude='rl_ladder/results' --exclude='rl_ladder/logs' --exclude='rl_ladder/plans' \
    --exclude='rl_ladder/dist' --exclude='rl_ladder/.venv' --exclude='__pycache__' \
    --exclude='*.pyc' --exclude='.pytest_cache' --exclude='.DS_Store' \
    rl_ladder
echo "Created $out"
echo "Upload:  scp $out <user>@firebird.swarthmore.edu:~/"
echo "Unpack:  tar -xzf ~/$(basename "$out") -C ~/"
