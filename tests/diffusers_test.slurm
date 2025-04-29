#!/bin/bash
#SBATCH --job-name=diffusers_cached      # job name
#SBATCH --account=commons                # or your condo account
#SBATCH --partition=commons              # has GPU nodes
#SBATCH --gres=gpu:1                     # request 1 GPU
#SBATCH --ntasks=1                       # single process
#SBATCH --cpus-per-task=4                # Python threading
#SBATCH --mem=16G                        # RAM per node
#SBATCH --time=00:30:00                  # 30-minute limit
#SBATCH --output=logs/diffusers_%j.out # stdout
#SBATCH --error=logs/diffusers_%j.err  # stderr

module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2
source ~/environments/comp646/bin/activate

USER_SCRATCH=$SHARED_SCRATCH/$USER
export HF_HOME=$USER_SCRATCH/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

JOBDIR = $USER_SCRATCH/$SLURM_JOB_ID
mkdir -p $JOBDIR

cp $SLURM_SUBMIT_DIR/tests/test_diffusers.py $JOBDIR/
cd $JOBDIR

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

python test_diffusers.py

