#!/bin/bash -l
#SBATCH --job-name=PE_SEOBNR_event9
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --account=andreasb

set -eo pipefail
export PS1=${PS1:-"batch"}

RUNDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
SCRIPT="PE_SEOBNR_event9.py"
POP_OUTDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outdir_population_exactfd"

module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate GW

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

for _k in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS NUMEXPR_NUM_THREADS VECLIB_MAXIMUM_THREADS BLIS_NUM_THREADS; do
    export "${_k}=1"
done

mkdir -p "${RUNDIR}"
cd "${RUNDIR}"

EVENT_INDEX=1
EVENT_NAME=$(printf "event_%04d" "${EVENT_INDEX}")
OUTDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outdir_population_run_SEOBNR"
LABEL="bns_${EVENT_NAME}_seobnr"

ZERO_NOISE=0
WIDEN_MC=0.0002
NLIVE=1000
DELTA_SIGMA=1.0
PE_NPOOL="${SLURM_CPUS_PER_TASK:-1}"

RW_NPOOL=4
RW_N_CHECKPOINT=2000
RW_METHOD="weighted"
RW_CLIP_WEIGHTS=0
RW_CLIP_MIN_LOG=-745
SKY_FRAME="detector"

RESUME_DIR="${TMPDIR:-${OUTDIR}}"
mkdir -p "${RESUME_DIR}"
RW_RESUME_FILE="${RESUME_DIR}/${LABEL}_reweighted_weights_resume.npz"

echo "================================================="
echo "Time: $(date)"
echo "Node: ${SLURM_NODELIST:-unknown}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "Event: ${EVENT_INDEX} (${EVENT_NAME})"
echo "Script: ${SCRIPT}"
echo "OUTDIR: ${OUTDIR}"
echo "LABEL: ${LABEL}"
echo "NLIVE: ${NLIVE}"
echo "RW_METHOD: ${RW_METHOD}"
echo "RW_CLIP_WEIGHTS: ${RW_CLIP_WEIGHTS}"
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
    --delta-sigma "${DELTA_SIGMA}"
    --npool "${PE_NPOOL}"
    --rw-npool "${RW_NPOOL}"
    --rw-checkpoint "${RW_N_CHECKPOINT}"
    --rw-resume-file "${RW_RESUME_FILE}"
    --rw-method "${RW_METHOD}"
    --rw-clip-min-log "${RW_CLIP_MIN_LOG}"
    --sky-frame "${SKY_FRAME}"
)

if [ "${ZERO_NOISE}" = "1" ]; then
    ARGS+=(--zero-noise)
fi

if [ "${RW_CLIP_WEIGHTS}" = "1" ]; then
    ARGS+=(--rw-clip-weights)
fi

srun -n 1 --cpus-per-task="${PE_NPOOL}" --cpu-bind=cores \
    python "${SCRIPT}" "${ARGS[@]}"