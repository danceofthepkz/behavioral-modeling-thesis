#!/bin/bash -l
# Summarise a plan once its array job has finished. Submitted by submit.sh.
#SBATCH --job-name=rl-ladder-collect
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit with sbatch from the rl_ladder folder}"
source scripts/activate.sh
python -m rl_ladder collect "${1:?plan path}" || {
    echo "Incomplete: resubmit the same experiment (finished units are skipped)," \
         "or run: python -m rl_ladder collect $1 --allow-partial"
    exit 1
}
