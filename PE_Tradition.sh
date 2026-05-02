#!/bin/bash -l
#SBATCH --job-name=PE_Tradition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=%x-%A_%a.out
#SBATCH --error=%x-%A_%a.err
#SBATCH --hint=nomultithread
#SBATCH --account=andreasb
##SBATCH --array=1-10

set -eo pipefail
export PS1=${PS1:-"batch"}

RUNDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
SCRIPT="PE_Traditional.py"
POP_OUTDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outdir_population_exactfd"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    EVENT_INDEX="${SLURM_ARRAY_TASK_ID}"
else
    EVENT_INDEX="1"
fi

module purge
module load anaconda3/2025.12
eval "$(conda shell.bash hook)"
conda activate GW

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

mkdir -p "${RUNDIR}"
cd "${RUNDIR}"

# ======================
# Main PE settings
# ======================
ZERO_NOISE=0
FIX_TIDAL=0
WIDEN_MC=0.1
NLIVE=200
NACCEPT=60
PE_NPOOL="${SLURM_CPUS_PER_TASK:-1}"

# ======================
# Traditional tidal prior
# ======================
LAMBDA_TILDE_MIN=10
LAMBDA_TILDE_MAX=10000
DELTA_LAMBDA_TILDE_MAX=3000

# ======================
# Labels / output
# ======================
EVENT_NAME=$(printf "event_%04d" "${EVENT_INDEX}")
OUTDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outdir_population_run_traditional"
LABEL="bns_${EVENT_NAME}_traditional_Mc0.1"

# ======================
# Reweight settings
# ======================
RW_NPOOL=4
RW_N_CHECKPOINT=2000
RW_METHOD="weighted"          # options: weighted, rejection, systematic
RW_USE_NESTED_SAMPLES=0       # 1 -> pass --rw-use-nested-samples
RESUME_DIR="${TMPDIR:-${OUTDIR}}"
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
echo "FIX_TIDAL=${FIX_TIDAL}"
echo "WIDEN_MC=${WIDEN_MC}"
echo "NLIVE=${NLIVE}"
echo "NACCEPT=${NACCEPT}"
echo "PE_NPOOL=${PE_NPOOL}"
echo "LAMBDA_TILDE_MIN=${LAMBDA_TILDE_MIN}"
echo "LAMBDA_TILDE_MAX=${LAMBDA_TILDE_MAX}"
echo "DELTA_LAMBDA_TILDE_MAX=${DELTA_LAMBDA_TILDE_MAX}"
echo "RW_NPOOL=${RW_NPOOL}"
echo "RW_N_CHECKPOINT=${RW_N_CHECKPOINT}"
echo "RW_METHOD=${RW_METHOD}"
echo "RW_USE_NESTED_SAMPLES=${RW_USE_NESTED_SAMPLES}"
echo "RW_RESUME_FILE=${RW_RESUME_FILE}"
echo "================================================="

python --version
which python
ls -l "${SCRIPT}"

ARGS=(
  --pop-outdir "${POP_OUTDIR}"
  --event-index "${EVENT_INDEX}"
  --outdir "${OUTDIR}"
  --label "${LABEL}"
  --widen-mc "${WIDEN_MC}"
  --nlive "${NLIVE}"
  --naccept "${NACCEPT}"
  --npool "${PE_NPOOL}"
  --lambda-tilde-min "${LAMBDA_TILDE_MIN}"
  --lambda-tilde-max "${LAMBDA_TILDE_MAX}"
  --delta-lambda-tilde-max "${DELTA_LAMBDA_TILDE_MAX}"
  --rw-npool "${RW_NPOOL}"
  --rw-checkpoint "${RW_N_CHECKPOINT}"
  --rw-resume-file "${RW_RESUME_FILE}"
  --rw-method "${RW_METHOD}"
)

if [ "${ZERO_NOISE}" = "1" ]; then
  ARGS+=(--zero-noise)
fi

if [ "${FIX_TIDAL}" = "1" ]; then
  ARGS+=(--fix-tidal)
fi

if [ "${RW_USE_NESTED_SAMPLES}" = "1" ]; then
  ARGS+=(--rw-use-nested-samples)
fi

srun -n 1 --cpus-per-task="${PE_NPOOL}" --cpu-bind=cores \
  python "${SCRIPT}" "${ARGS[@]}"
