#!/bin/bash -l
#SBATCH --job-name=population_exactsignal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=9:00:00
#SBATCH --output=/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/logs/%x-%j.out
#SBATCH --error=/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/logs/%x-%j.err
#SBATCH --account=andreasb

set -eo pipefail
export PS1=${PS1:-"batch"}

RUNDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
SCRIPT="scripts/Injection_population_eosfit.py"

module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate GW

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

TARGET_N=100
BASE_SEED=123
PROGRESS_EVERY=50
OUTDIR="outputs/outdir_population_exactfd_mband0"
LABEL="bns_pop_inj_exactfd_mband0"

mkdir -p "${RUNDIR}"
cd "${RUNDIR}"

echo "Time: $(date)"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
python --version
which python
ls -l "${SCRIPT}"

python "${SCRIPT}" \
  --outdir "${OUTDIR}" \
  --label "${LABEL}" \
  --target-n "${TARGET_N}" \
  --base-seed "${BASE_SEED}" \
  --progress-every "${PROGRESS_EVERY}"
