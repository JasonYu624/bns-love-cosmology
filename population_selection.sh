#!/bin/bash -l
#SBATCH --job-name=population_selection_eosfit
#SBATCH --array=0-63%64
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=andreasb

set -eo pipefail
export PS1=${PS1:-"batch"}

RUNDIR="/scratch/gpfs/ANDREASB/fy6204/GW/Workspace"
SCRIPT="population_selection_bias_eosfit.py"

module purge
module load anaconda3/2025.6
eval "$(conda shell.bash hook)"
conda activate GW

export N_WORKERS="${N_WORKERS:-64}"
export TARGET_SUCCESS="${TARGET_SUCCESS:-5120}"

export CHUNK="${CHUNK:-512}"
export PROGRESS_EVERY_DET="${PROGRESS_EVERY_DET:-512}"

export BILBY_SEED="${BILBY_SEED:-123}"
export OUTDIR="${OUTDIR:-outdir_selection_calib_eosfit}"
export LABEL="${LABEL:-bns_sel_calib_eosfit}"

# Selection proposal. Keep this flat over the PE/hierarchy source-mass support
# so the VT proposal covers the full recycling target support.
export MASS_DIST="${MASS_DIST:-flat}"

# Match PE_eosfit_reweight.py source-frame mass constraints.
export M_MIN="${M_MIN:-0.8}"
export M_MAX="${M_MAX:-1.8}"
# 只有在 MASS_DIST=gaussian 时才会用到
# export M_MU="${M_MU:-1.33}"
# export M_SIGMA="${M_SIGMA:-0.09}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mkdir -p "${RUNDIR}"
cd "${RUNDIR}"

echo "Time: $(date)"
echo "Node list: ${SLURM_NODELIST:-unknown}"
echo "CPUs per task: ${SLURM_CPUS_PER_TASK:-unknown}"
echo "SCRIPT=${SCRIPT}"
echo "OUTDIR=${OUTDIR}"
echo "LABEL=${LABEL}"
echo "MASS_DIST=${MASS_DIST}"
echo "M_MIN=${M_MIN}"
echo "M_MAX=${M_MAX}"
echo "TARGET_SUCCESS=${TARGET_SUCCESS}"
echo "CHUNK=${CHUNK}"
python --version
which python
ls -l "${SCRIPT}"

python "${SCRIPT}"
