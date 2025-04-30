module purge
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1

# Ensure scratch I/O directory exists
mkdir $SHARED_SCRATCH/$USER

# Initialize session environment
TEMP_VENV=gpuenv_tmp
python -m venv $SHARED_SCRATCH/$USER/$TEMP_VENV
source $SHARED_SCRATCH/$USER/$TEMP_VENV/bin/activate

# Set pip cache directory
export PIP_CACHE_DIR=$SHARED_SCRATCH/$USER/.cache/pip
mkdir $PIP_CACHE_DIR

pip install bitsandbytes transformers accelerate peft
