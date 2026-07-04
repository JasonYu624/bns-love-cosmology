#!/bin/bash -l
#SBATCH --job-name=HyperPE_SEOBNR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
##SBATCH --constraint="nomig&gpu40"
#SBATCH --partition=mig
#SBATCH --gres=gpu:1
#SBATCH --account=andreasb
#SBATCH --output=/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/logs/%x-%j.err
#SBATCH --hint=nomultithread

set -eo pipefail
export PS1=${PS1:-"batch"}

RUNDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
SCRIPT="scripts/HyperPE_eosfit.py"

PE_OUTDIR="${RUNDIR}/outputs/outdir_population_run_SEOBNR"
POSTERIOR_GLOB="${PE_OUTDIR}/*_reweighted_posterior_augmented.csv"

VT_GLOB="${RUNDIR}/outputs/outdir_selection_calib_SEOBNR/worker_*/detected_chunk*.npz"

OUTDIR="${RUNDIR}/outputs/outdir_hyperpe_run_SEOBNR_10"
LABEL="hyperpe_SEOBNR_10"

NLIVE=500
SEED=12345
BACKEND="jax"
# EVENT_LIMIT:
#   10 = first 10 events
#   50 = first 50 events
#   0  = all matched events
EVENT_LIMIT=10
module purge
module load anaconda3/2025.12
conda activate GW_gpu

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Backend settings
export JAX_ENABLE_X64=True
export WCOSMO_ARRAY_API=jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Keep CPU thread pools controlled.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

mkdir -p "${RUNDIR}" "${OUTDIR}"
cd "${RUNDIR}"

echo "================================================="
echo "Time: $(date)"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unknown}"
echo "RUNDIR=${RUNDIR}"
echo "SCRIPT=${SCRIPT}"
echo "PE_OUTDIR=${PE_OUTDIR}"
echo "POSTERIOR_GLOB=${POSTERIOR_GLOB}"
echo "EVENT_LIMIT=${EVENT_LIMIT}"
echo "VT_GLOB=${VT_GLOB}"
echo "OUTDIR=${OUTDIR}"
echo "LABEL=${LABEL}"
echo "NLIVE=${NLIVE}"
echo "SEED=${SEED}"
echo "BACKEND=${BACKEND}"
echo "================================================="

python --version
which python
ls -l "${SCRIPT}"

echo "Matched posterior files:"
mapfile -t POSTERIOR_FILES < <(find "${PE_OUTDIR}" -maxdepth 1 -name "*_reweighted_posterior_augmented.csv" | sort)

if [[ "${#POSTERIOR_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no posterior files matched ${POSTERIOR_GLOB}" >&2
  exit 1
fi

printf '%s\n' "${POSTERIOR_FILES[@]}"
echo "Number of posterior files: ${#POSTERIOR_FILES[@]}"

echo "Checking VT files:"
mapfile -t VT_FILES < <(find "${RUNDIR}/outputs/outdir_selection_calib_SEOBNR" -path "*/worker_*/*" -name "detected_chunk*.npz" | sort)
# mapfile -t VT_FILES < <(find "${RUNDIR}/outputs/outdir_selection_calib_eosfit" -path "*/worker_*/*" -name "detected_chunk*.npz" | sort)

if [[ "${#VT_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no VT files matched ${VT_GLOB}" >&2
  exit 1
fi

echo "Number of VT files: ${#VT_FILES[@]}"

ARGS=(
  --posterior-glob "${POSTERIOR_GLOB}"
  --max-events "${EVENT_LIMIT}"
  --vt-glob "${VT_GLOB}"
  --outdir "${OUTDIR}"
  --label "${LABEL}"
  --nlive "${NLIVE}"
  --backend "${BACKEND}"
  --jit-likelihood
  --seed "${SEED}"
  --no-resume
)

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun -n 1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=cores \
    python "${SCRIPT}" "${ARGS[@]}"
else
  python "${SCRIPT}" "${ARGS[@]}"
fi

echo "================================================="
echo "Finished at: $(date)"
echo "Diagnostics: ${OUTDIR}/${LABEL}_diagnostics.json"
echo "Variance summary: ${OUTDIR}/${LABEL}_variance_components_summary.csv"
echo "Variance correlations: ${OUTDIR}/${LABEL}_variance_correlations_by_parameter.csv"
echo "================================================="
