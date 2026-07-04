#!/bin/bash -l
#SBATCH --job-name=HyperPE_Spectral_SEOBNR_ev54_63
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
SCRIPT="scripts/HyperPE_spectral_SEOBNR.py"

# Events 54-63: contiguous 10-event block staged as symlinks (selected because it
# shows the q-underestimation and dL-overestimation biases; the first 10 events are
# prior-dominated).  The HyperPE script selects events via sorted(glob)[:max_events],
# so we point the glob at the staging dir and assert exactly 10 with EVENT_LIMIT=10.
PE_OUTDIR="${RUNDIR}/outputs/hyperpe_inputs_SEOBNR_ev54_63"
POSTERIOR_GLOB="${PE_OUTDIR}/*_reweighted_posterior_augmented.csv"

VT_BASE="${RUNDIR}/outputs/outdir_selection_calib_SEOBNR"
VT_GLOB="${VT_BASE}/worker_*/detected_chunk*.npz"

OUTDIR="${RUNDIR}/outputs/outdir_spectral_siren_seobnr_ev54_63"
LABEL="spectral_siren_seobnr_ev54_63"

NLIVE=500
SEED=12345
EVENT_LIMIT=10
POSTPROCESS_MAX_SAMPLES=5000
RESUME_FLAG="--no-resume"

module purge
module load anaconda3/2025.12
conda activate GW_gpu

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

export JAX_ENABLE_X64=True
export WCOSMO_ARRAY_API=jax
export XLA_PYTHON_CLIENT_PREALLOCATE=false

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
echo "VT_GLOB=${VT_GLOB}"
echo "OUTDIR=${OUTDIR}"
echo "LABEL=${LABEL}"
echo "NLIVE=${NLIVE}"
echo "SEED=${SEED}"
echo "EVENT_LIMIT=${EVENT_LIMIT}"
echo "POSTPROCESS_MAX_SAMPLES=${POSTPROCESS_MAX_SAMPLES}"
echo "RESUME_FLAG=${RESUME_FLAG}"
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

if [[ "${#POSTERIOR_FILES[@]}" -lt "${EVENT_LIMIT}" ]]; then
  echo "ERROR: requested EVENT_LIMIT=${EVENT_LIMIT}, but only found ${#POSTERIOR_FILES[@]} posterior files." >&2
  exit 1
fi

printf '%s\n' "${POSTERIOR_FILES[@]:0:${EVENT_LIMIT}}"
echo "Number of posterior files found: ${#POSTERIOR_FILES[@]}"
echo "Number of posterior files selected: ${EVENT_LIMIT}"

echo "Checking VT files:"
mapfile -t VT_FILES < <(find "${VT_BASE}" -path "*/worker_*/*" -name "detected_chunk*.npz" | sort)

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
  --seed "${SEED}"
  --postprocess-max-samples "${POSTPROCESS_MAX_SAMPLES}"
  "${RESUME_FLAG}"
)

echo "================================================="
echo "Starting spectral-siren SEOBNR run with events 54-63 (${EVENT_LIMIT} events)"
echo "================================================="

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun -n 1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=cores \
    python "${SCRIPT}" "${ARGS[@]}"
else
  python "${SCRIPT}" "${ARGS[@]}"
fi

echo "================================================="
echo "Finished at: $(date)"
echo "Corner: ${OUTDIR}/corner_${LABEL}.png"
echo "Posterior with variance: ${OUTDIR}/${LABEL}_posterior_with_variance.csv"
echo "Variance summary: ${OUTDIR}/${LABEL}_variance_components_summary.csv"
echo "Variance correlations: ${OUTDIR}/${LABEL}_variance_correlations_by_parameter.csv"
echo "================================================="
