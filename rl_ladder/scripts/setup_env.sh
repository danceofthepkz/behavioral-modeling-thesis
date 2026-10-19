#!/usr/bin/env bash
# Create the Python environment.
#   scripts/setup_env.sh conda    conda env "rl-ladder" from environment.yml (Firebird)
#   scripts/setup_env.sh venv     <project>/.venv from requirements.txt (any machine)
#   scripts/setup_env.sh mice-rl  add the two missing packages to the existing
#                                 Firebird env "mice-rl" (then: export RL_LADDER_ENV=mice-rl)
set -eo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"
mode="${1:-}"
case "$mode" in
  conda)
    command -v conda >/dev/null || { echo "conda not found (module load / miniforge?)"; exit 1; }
    name="${RL_LADDER_ENV:-rl-ladder}"
    if conda env list | awk '{print $1}' | grep -qx "$name"; then
      conda env update -n "$name" -f environment.yml --prune
    else
      conda env create -n "$name" -f environment.yml
    fi
    echo "Done. Next: source scripts/activate.sh   (RL_LADDER_ENV=$name)"
    ;;
  venv)
    py="${PYTHON:-python3.11}"
    command -v "$py" >/dev/null || { echo "$py not found; set PYTHON=/path/to/python3.11"; exit 1; }
    "$py" -m venv .venv
    .venv/bin/python -m pip install --upgrade pip
    .venv/bin/python -m pip install -r requirements.txt
    echo "Done. Next: source scripts/activate.sh"
    ;;
  mice-rl)
    # Reuse the activation script that already works on Firebird for rl_firebird_v2.
    if [[ -f "$HOME/rl_firebird_v2/activate_rl.sh" ]]; then
      source "$HOME/rl_firebird_v2/activate_rl.sh"
    else
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "${RL_LADDER_ENV:-$HOME/.conda/envs/mice-rl}"
    fi
    echo "Installing into: $(command -v python)"
    python -m pip install "PyYAML==6.0.3" "pytest==9.1.1"
    python -c "import numpy, scipy, numba, pandas, yaml, pytest; print('environment OK')"
    echo "Done. Next: export RL_LADDER_ENV=\$HOME/.conda/envs/mice-rl; source scripts/activate.sh"
    ;;
  *)
    sed -n '2,7p' "$0"; exit 1 ;;
esac
