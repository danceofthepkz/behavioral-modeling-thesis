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
    echo "Collection failed: inspect collection_status.json and the task's *_quality.json in the run's summary."
    echo "Missing units: resubmit the same experiment; finished units are skipped."
    echo "Failed CV quality: inspect cv_verification.csv; change fit settings in a new run if needed."
    echo "Failed recovery quality: inspect recovery_verification.csv and recovery_quality.json; matrices are withheld."
    exit 1
}
