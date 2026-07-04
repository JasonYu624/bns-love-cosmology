#!/bin/bash -l
#SBATCH --job-name=PE_SEOBNR_rw
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/logs/%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --account=andreasb
##SBATCH --array=93-100

set -eo pipefail
export PS1=${PS1:-"batch"}

RUNDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
SCRIPT="scripts/PE_SEOBNR_reweight.py"

POP_OUTDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outputs/outdir_population_SEOBNR"
OUTDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outputs/outdir_population_run_SEOBNR"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    EVENT_INDEX="${SLURM_ARRAY_TASK_ID}"
else
    EVENT_INDEX="99"
fi

module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate GW
# conda activate GW_nodyn

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

LAL_DATA_PATH="/scratch/gpfs/ANDREASB/lalsuite-extra/lalsuite-waveform-data/waveform_data"
export LAL_DATA_PATH

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

mkdir -p "${RUNDIR}" "${OUTDIR}"
cd "${RUNDIR}"

# ======================
# Main PE settings
# ======================
ZERO_NOISE=0
WIDEN_MC=0.0002
NLIVE=1000
DELTA_SIGMA=1.0
PE_NPOOL="${SLURM_CPUS_PER_TASK:-1}"

# ======================
# Labels / output
# ======================
EVENT_NAME=$(printf "event_%04d" "${EVENT_INDEX}")
LABEL="bns_${EVENT_NAME}_seobnr"

# ======================
# Reweight settings
# posterior only
# ======================
RW_NPOOL=4
RW_N_CHECKPOINT=2000
RW_METHOD="weighted"
RW_USE_NESTED_SAMPLES=0
SKY_FRAME="detector"

# Use a persistent directory, not TMPDIR, so the resume file is inspectable.
# The Python script will remove stale resume files unless --resume-reweight is passed.
RESUME_DIR="${OUTDIR}/reweight_resume"
mkdir -p "${RESUME_DIR}"
RW_RESUME_FILE="${RESUME_DIR}/${LABEL}_reweighted_weights_resume.npz"

echo "================================================="
echo "Time: $(date)"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Running Event Index: ${EVENT_INDEX}"
echo "RUNDIR=${RUNDIR}"
echo "SCRIPT=${SCRIPT}"
echo "POP_OUTDIR=${POP_OUTDIR}"
echo "OUTDIR=${OUTDIR}"
echo "LABEL=${LABEL}"
echo "ZERO_NOISE=${ZERO_NOISE}"
echo "WIDEN_MC=${WIDEN_MC}"
echo "NLIVE=${NLIVE}"
echo "DELTA_SIGMA=${DELTA_SIGMA}"
echo "PE_NPOOL=${PE_NPOOL}"
echo "RW_NPOOL=${RW_NPOOL}"
echo "RW_N_CHECKPOINT=${RW_N_CHECKPOINT}"
echo "RW_METHOD=${RW_METHOD}"
echo "RW_USE_NESTED_SAMPLES=${RW_USE_NESTED_SAMPLES}"
echo "SKY_FRAME=${SKY_FRAME}"
echo "RW_RESUME_FILE=${RW_RESUME_FILE}"
echo "LAL_DATA_PATH=${LAL_DATA_PATH}"
echo "TMPDIR=${TMPDIR:-not_set}"
echo "================================================="

python --version
which python
ls -l "${SCRIPT}"
ls -ld "${POP_OUTDIR}"
ls -ld "${OUTDIR}"

ARGS=(
  --pop-outdir "${POP_OUTDIR}"
  --event-index "${EVENT_INDEX}"
  --outdir "${OUTDIR}"
  --label "${LABEL}"
  --widen-mc "${WIDEN_MC}"
  --nlive "${NLIVE}"
  --delta-sigma "${DELTA_SIGMA}"
  --npool "${PE_NPOOL}"
  --rw-npool "${RW_NPOOL}"
  --rw-checkpoint "${RW_N_CHECKPOINT}"
  --rw-resume-file "${RW_RESUME_FILE}"
  --rw-method "${RW_METHOD}"
  --sky-frame "${SKY_FRAME}"
)

if [ "${ZERO_NOISE}" = "1" ]; then
  ARGS+=(--zero-noise)
fi

if [ "${RW_USE_NESTED_SAMPLES}" = "1" ]; then
  ARGS+=(--rw-use-nested-samples)
fi

srun -n 1 --cpus-per-task="${PE_NPOOL}" --cpu-bind=cores \
  python "${SCRIPT}" "${ARGS[@]}"
