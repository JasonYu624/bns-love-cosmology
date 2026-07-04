#!/bin/bash -l
#SBATCH --job-name=HyperPE_7ev
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

PE_OUTDIR="${RUNDIR}/outdir_population_run"
VT_GLOB="${RUNDIR}/outdir_selection_calib_eosfit/worker_*/detected_chunk*.npz"
INPUT7="${RUNDIR}/hyperpe_input_7events"
OUTDIR="${RUNDIR}/outdir_hyperpe_run_7ev"
LABEL="hyperpe_7ev_jit"

NLIVE=500

BACKEND="jax"

module purge
module load anaconda3/2025.12
# module load anaconda3/2025.6 || true
# if ! command -v conda >/dev/null 2>&1; then
#   source /usr/licensed/anaconda3/2025.6/etc/profile.d/conda.sh
# fi
# eval "$(conda shell.bash hook)"
conda activate GW_gpu

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

# Backend settings
# export JAX_ENABLE_X64=True
# export WCOSMO_ARRAY_API=jax
# export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Keep CPU thread pools controlled.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

mkdir -p "${RUNDIR}" "${OUTDIR}" "${INPUT7}"
cd "${RUNDIR}"

# Rebuild strict input directory.
rm -f "${INPUT7}"/*_reweighted_posterior_augmented.csv

ln -sf "${PE_OUTDIR}/bns_event_0002_eosfit_dyn_reweighted_posterior_augmented.csv" "${INPUT7}/"
ln -sf "${PE_OUTDIR}/bns_event_0003_eosfit_dyn_reweighted_posterior_augmented.csv" "${INPUT7}/"
ln -sf "${PE_OUTDIR}/bns_event_0005_eosfit_dyn_reweighted_posterior_augmented.csv" "${INPUT7}/"
ln -sf "${PE_OUTDIR}/bns_event_0006_eosfit_dyn_reweighted_posterior_augmented.csv" "${INPUT7}/"
ln -sf "${PE_OUTDIR}/bns_event_0007_eosfit_dyn_reweighted_posterior_augmented.csv" "${INPUT7}/"
ln -sf "${PE_OUTDIR}/bns_event_0008_eosfit_dyn_reweighted_posterior_augmented.csv" "${INPUT7}/"
ln -sf "${PE_OUTDIR}/bns_event_0010_eosfit_dyn_reweighted_posterior_augmented.csv" "${INPUT7}/"

echo "================================================="
echo "Time: $(date)"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unknown}"
echo "RUNDIR=${RUNDIR}"
echo "SCRIPT=${SCRIPT}"
echo "INPUT7=${INPUT7}"
echo "VT_GLOB=${VT_GLOB}"
echo "OUTDIR=${OUTDIR}"
echo "LABEL=${LABEL}"
echo "NLIVE=${NLIVE}"
echo "BACKEND=${BACKEND}"
echo "================================================="

python --version
which python
ls -l "${SCRIPT}"
ls -1 "${INPUT7}"/*_reweighted_posterior_augmented.csv

ARGS=(
  --posterior-glob "${INPUT7}/*_reweighted_posterior_augmented.csv"
  --vt-glob "${VT_GLOB}"
  --outdir "${OUTDIR}"
  --label "${LABEL}"
  --nlive "${NLIVE}"
  --backend "${BACKEND}"
  --jit-likelihood
  --seed "12345"
  --no-resume
)

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  srun -n 1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-4}" --cpu-bind=cores \
    python "${SCRIPT}" "${ARGS[@]}"
else
  python "${SCRIPT}" "${ARGS[@]}"
fi

echo "================================================="
echo "Finished at: $(date)"
echo "Diagnostics: ${OUTDIR}/${LABEL}_diagnostics.json"
echo "================================================="
