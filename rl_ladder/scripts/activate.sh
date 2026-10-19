# Source this file (bash or zsh) before running anything:
#     source scripts/activate.sh
#
# It activates the Python environment and exports the RL_LADDER_* variables.
# Environment choice, first match wins:
#   1. $RL_LADDER_VENV                 an existing virtualenv
#   2. <project>/.venv                 created by `scripts/setup_env.sh venv`
#   3. <project>/../../.venv           the thesis workspace venv (local Mac layout)
#   4. conda env $RL_LADDER_ENV        default name: rl-ladder (Firebird)
# Data: $RL_LADDER_DATA, else the first folder containing GS027.npz among
#   <project>/../../AuditoryCategorizationSwat/thesis/rl_firebird_v2/data   (local)
#   $HOME/rl_firebird_v2/data                                               (Firebird)

_rl_src="${BASH_SOURCE[0]:-${(%):-%x}}"
export RL_LADDER_HOME="$(cd "$(dirname "$_rl_src")/.." && pwd)"
unset _rl_src

_rl_activate() {
    local venv=""
    for cand in "${RL_LADDER_VENV:-}" "$RL_LADDER_HOME/.venv" "$RL_LADDER_HOME/../../.venv"; do
        if [[ -n "$cand" && -f "$cand/bin/activate" ]]; then venv="$cand"; break; fi
    done
    if [[ -n "$venv" ]]; then
        source "$venv/bin/activate" || return 1
        return 0
    fi
    if [[ -n "${RL_CONDA_INIT:-}" ]]; then
        source "$RL_CONDA_INIT" || return 1
    elif command -v conda >/dev/null 2>&1; then
        source "$(conda info --base)/etc/profile.d/conda.sh" || return 1
    elif [[ -r "$HOME/miniforge3/etc/profile.d/conda.sh" ]]; then
        source "$HOME/miniforge3/etc/profile.d/conda.sh" || return 1
    else
        echo "rl_ladder: no virtualenv and no conda found. Run scripts/setup_env.sh first," \
             "or set RL_LADDER_VENV / RL_CONDA_INIT." >&2
        return 1
    fi
    conda activate "${RL_LADDER_ENV:-rl-ladder}" || {
        echo "rl_ladder: conda env '${RL_LADDER_ENV:-rl-ladder}' not found;" \
             "run scripts/setup_env.sh conda" >&2
        return 1
    }
}
_rl_activate || return 1
unset -f _rl_activate

export PYTHONPATH="$RL_LADDER_HOME${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
       NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0 MPLBACKEND=Agg

if [[ -z "${RL_LADDER_DATA:-}" ]]; then
    for _rl_cand in "$RL_LADDER_HOME/../../AuditoryCategorizationSwat/thesis/rl_firebird_v2/data" \
                    "$HOME/rl_firebird_v2/data"; do
        if [[ -f "$_rl_cand/GS027.npz" ]]; then
            export RL_LADDER_DATA="$(cd "$_rl_cand" && pwd)"
            break
        fi
    done
    unset _rl_cand
fi

if ! python -c "import numpy, scipy, numba, pandas, yaml" >/dev/null 2>&1; then
    echo "rl_ladder: missing packages in $(command -v python); see requirements.txt" >&2
fi
echo "rl_ladder: python=$(command -v python)"
echo "rl_ladder: RL_LADDER_DATA=${RL_LADDER_DATA:-<not set: export it before fitting>}"
echo "rl_ladder: RL_LADDER_RESULTS=${RL_LADDER_RESULTS:-$RL_LADDER_HOME/results}"
