#!/bin/bash -l
# One array index = `chunk` consecutive units of a plan. Submitted by submit.sh.
#   sbatch --array=0-N scripts/slurm/unit.sbatch PLAN CHUNK
#SBATCH --job-name=rl-ladder
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the rl_ladder folder}"
source scripts/activate.sh
# Per-job numba cache: avoids concurrent writes to a shared cache directory.
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/rl-ladder-numba-${USER}-${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
plan="${1:?plan path}"; chunk="${2:-1}"
echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID} node=$(hostname) plan=${plan} chunk=${chunk}"
srun python -m rl_ladder unit "$plan" "$SLURM_ARRAY_TASK_ID" --chunk-size "$chunk"
