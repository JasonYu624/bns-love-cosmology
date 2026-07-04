#!/bin/bash -l
#SBATCH --job-name=VTinjections_SEOBNR
#SBATCH --array=0-63%64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/logs/%x-%j.err
#SBATCH --account=andreasb

set -eo pipefail

RUNDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
SCRIPT="scripts/VTinjections_SEOBNR.py"
LAL_DATA_PATH="/scratch/gpfs/ANDREASB/lalsuite-extra/lalsuite-waveform-data/waveform_data"

module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate GW

export LAL_DATA_PATH
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

N_WORKERS=64
TARGET_SUCCESS=5120
CHUNK=512
PROGRESS_EVERY_DET=512
BASE_SEED=123
OUTDIR="outputs/outdir_selection_calib_SEOBNR"
LABEL="bns_sel_calib_SEOBNR"
WORKER_ID="${SLURM_ARRAY_TASK_ID:-0}"

cd "${RUNDIR}"

echo "Time: $(date)"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "Array task ID: ${WORKER_ID}"
echo "LAL_DATA_PATH: ${LAL_DATA_PATH}"
python --version
which python
ls -l "${SCRIPT}"

python "${SCRIPT}" \
  --outdir "${OUTDIR}" \
  --label "${LABEL}" \
  --target-success "${TARGET_SUCCESS}" \
  --base-seed "${BASE_SEED}" \
  --worker-id "${WORKER_ID}" \
  --n-workers "${N_WORKERS}" \
  --chunk "${CHUNK}" \
  --progress-every-det "${PROGRESS_EVERY_DET}"
