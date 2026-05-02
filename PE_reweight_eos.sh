#!/bin/bash -l
#SBATCH --job-name=PE_reweight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --hint=nomultithread
#SBATCH --account=andreasb

set -eo pipefail
export PS1=${PS1:-"batch"}

RUNDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
SCRIPT="PE_reweight_eosfit.py"

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

# ----------------------
# User-configurable envs
# ----------------------
export EVENT_INDEX=1
EVENT_NAME=$(printf "event_%04d" "${EVENT_INDEX}")

export POP_OUTDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outdir_population_exactfd"
export OUTDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace/outdir_population_run_test"

export ZERO_NOISE=0
export WIDEN_MC=0.1
export UPDATE_FIDUCIAL=0
export Q_PRIOR="uniform_reflected"

# keep LABEL consistent with WIDEN_MC
export LABEL="bns_${EVENT_NAME}_UR_eosfit_exactsignal_Mc0.1"
export RW_LABEL="${LABEL}_reweighted_full"

export PRIOR_FILE="BNS_spin.prior"

# ----------------------
# Reweight settings
# ----------------------
export BILBY_RW_NPOOL=8
export RW_N_CHECKPOINT=2000
export RW_USE_NESTED_SAMPLES=0
export RW_CLEAN_RESUME=1

# Put resume/checkpoint files on local disk if available
RESUME_DIR="${TMPDIR:-${OUTDIR}}"
mkdir -p "${RESUME_DIR}"
export RW_RESUME_FILE="${RESUME_DIR}/${RW_LABEL}_weights_resume_test_Mc0.1.npz"

echo "Time: $(date)"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "RUNDIR=${RUNDIR}"
echo "SCRIPT=${SCRIPT}"
echo "POP_OUTDIR=${POP_OUTDIR}"
echo "OUTDIR=${OUTDIR}"
echo "EVENT_INDEX=${EVENT_INDEX}"
echo "LABEL=${LABEL}"
echo "RW_LABEL=${RW_LABEL}"
echo "PRIOR_FILE=${PRIOR_FILE}"
echo "ZERO_NOISE=${ZERO_NOISE}"
echo "WIDEN_MC=${WIDEN_MC}"
echo "UPDATE_FIDUCIAL=${UPDATE_FIDUCIAL}"
echo "Q_PRIOR=${Q_PRIOR}"
echo "BILBY_RW_NPOOL=${BILBY_RW_NPOOL}"
echo "RW_RESUME_FILE=${RW_RESUME_FILE}"
echo "RW_N_CHECKPOINT=${RW_N_CHECKPOINT}"
echo "RW_USE_NESTED_SAMPLES=${RW_USE_NESTED_SAMPLES}"
echo "RW_CLEAN_RESUME=${RW_CLEAN_RESUME}"
echo "TMPDIR=${TMPDIR:-not_set}"
python --version
which python
ls -l "${SCRIPT}"

srun python "${SCRIPT}"