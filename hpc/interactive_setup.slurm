module purge
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1

# Ensure scratch I/O directory exists
mkdir -p $SHARED_SCRATCH/$USER

# Initialize session environment
TEMP_VENV=gpuenv_$SLURM_JOB_ID
python -m venv $SHARED_SCRATCH/$USER/$TEMP_VENV
source $SHARED_SCRATCH/$USER/$TEMP_VENV/bin/activate

# Set pip cache directory
export PIP_CACHE_DIR=$SHARED_SCRATCH/$USER/.cache/pip
mkdir -p $PIP_CACHE_DIR

# Since git disallowed on compute nodes, clone from home

pip install bitsandbytes transformers accelerate peft
cp -r $HOME/git_to_compute/diffusers $SHARED_SCRATCH/$USER/$TEMP_VENV

