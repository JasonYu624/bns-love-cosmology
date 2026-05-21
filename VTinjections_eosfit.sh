#!/bin/bash -l
#SBATCH --job-name=VTinjections_eosfit
#SBATCH --array=0-63%64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=andreasb

set -eo pipefail

RUNDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
SCRIPT="VTinjections_eosfit.py"

module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate GW

N_WORKERS=64
TARGET_SUCCESS=5120
CHUNK=512
PROGRESS_EVERY_DET=512
BASE_SEED=123
OUTDIR="outdir_selection_calib_eosfit"
LABEL="bns_sel_calib_eosfit"
WORKER_ID="${SLURM_ARRAY_TASK_ID:-0}"

cd "${RUNDIR}"

python "${SCRIPT}" \
  --outdir "${OUTDIR}" \
  --label "${LABEL}" \
  --target-success "${TARGET_SUCCESS}" \
  --base-seed "${BASE_SEED}" \
  --worker-id "${WORKER_ID}" \
  --n-workers "${N_WORKERS}" \
  --chunk "${CHUNK}" \
  --progress-every-det "${PROGRESS_EVERY_DET}"
