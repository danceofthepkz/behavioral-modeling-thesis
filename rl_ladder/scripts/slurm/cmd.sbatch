#!/bin/bash -l
# Run any command on a compute node, e.g. the tests:
#   sbatch --partition=unowned scripts/slurm/cmd.sbatch scripts/test.sh
#SBATCH --job-name=rl-ladder-cmd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the rl_ladder folder}"
source scripts/activate.sh
export NUMBA_CACHE_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/rl-ladder-numba-${USER}-${SLURM_JOB_ID}"
mkdir -p "$NUMBA_CACHE_DIR"
[[ $# -gt 0 ]] || { echo "Supply a command"; exit 1; }
srun "$@"
