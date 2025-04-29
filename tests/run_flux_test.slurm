#!/bin/bash
#SBATCH --job-name=diffusers_cached
#SBATCH --account=commons
#SBATCH --partition=commons
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/diffusers_%j.out
#SBATCH --error=logs/diffusers_%j.err

module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1
source ~/environments/comp646/bin/activate

export HF_TOKEN=$(< $SLURM_SUBMIT_DIR/tests/.TOKEN)

USER_SCRATCH=$SHARED_SCRATCH/$USER
export HF_HOME=$USER_SCRATCH/.cache/huggingface
# export TRANSFORMERS_OFFLINE=1
# export HF_DATASETS_OFFLINE=1

JOBDIR=$USER_SCRATCH/$SLURM_JOB_ID
mkdir -p $JOBDIR

cp $SLURM_SUBMIT_DIR/tests/flux_test.py $JOBDIR/
cd $JOBDIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# python $JOBDIR/flux_test.py
accelerate launch --num_processes 4 $JOBDIR/flux_test.py
