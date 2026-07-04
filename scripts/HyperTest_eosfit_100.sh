#!/bin/bash -l
#SBATCH --job-name=HyperTest_EOSFIT_truth_100
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

PE_OUTDIR="${RUNDIR}/outdir_truth_delta_eosfit_SEOBNR"
POSTERIOR_GLOB="${PE_OUTDIR}/*_reweighted_posterior_augmented.csv"

VT_BASE="${RUNDIR}/outdir_selection_calib_SEOBNR"
VT_GLOB="${VT_BASE}/worker_*/detected_chunk*.npz"

EVENT_LIMIT=100

OUTDIR="${RUNDIR}/outdir_hyperpe_eosfit_truth_delta_${EVENT_LIMIT}"
LABEL="eosfit_truth_delta_${EVENT_LIMIT}"

NLIVE=500
SEED=12345
BACKEND="jax"

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

echo "Checking that ${SCRIPT} contains the VT total_generated fix:"
if ! grep -q "total_generated_naive_per_chunk_sum" "${SCRIPT}"; then
  echo "ERROR: ${SCRIPT} does not look like the vtfix version." >&2
  echo "Please copy the fixed script to ${RUNDIR}/${SCRIPT} before submitting." >&2
  exit 1
fi

echo "Matched truth-delta posterior files:"
mapfile -t POSTERIOR_FILES < <(find "${PE_OUTDIR}" -maxdepth 1 -name "*_reweighted_posterior_augmented.csv" | sort)

if [[ "${#POSTERIOR_FILES[@]}" -eq 0 ]]; then
  echo "ERROR: no posterior files matched ${POSTERIOR_GLOB}" >&2
  exit 1
fi

printf '%s\n' "${POSTERIOR_FILES[@]}"
echo "Number of posterior files: ${#POSTERIOR_FILES[@]}"

if [[ "${EVENT_LIMIT}" -gt 0 && "${#POSTERIOR_FILES[@]}" -lt "${EVENT_LIMIT}" ]]; then
  echo "ERROR: EVENT_LIMIT=${EVENT_LIMIT}, but only found ${#POSTERIOR_FILES[@]} posterior files." >&2
  exit 1
fi

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
  --backend "${BACKEND}"
  --jit-likelihood
  --seed "${SEED}"
  --no-resume
)

echo "Running command:"
printf ' %q' python "${SCRIPT}" "${ARGS[@]}"
echo

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
echo "Full extra statistics: ${OUTDIR}/${LABEL}_full_posterior_extra_statistics.csv"
echo "Corner: ${OUTDIR}/corner_${LABEL}.png"
echo "================================================="