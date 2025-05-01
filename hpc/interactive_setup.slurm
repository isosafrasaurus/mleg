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
cp $HOME/mleg/hpc/flux_inference_test.py $SHARED_SCRATCH/$USER/$TEMP_VENV
pip install bitsandbytes transformers accelerate peft
cp -r $HOME/git_to_compute/diffusers $SHARED_SCRATCH/$USER/$TEMP_VENV
pip install $SHARED_SCRATCH/$USER/$TEMP_VENV/diffusers
pip install sentencepiece

export HF_HOME=$SHARED_SCRATCH/$USER/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export DIFFUSERS_CACHE=$HF_HOME
mkdir -p $HF_HOME

TOKEN_FILE="$HOME/.HF_TOKEN"

if [ -f "$TOKEN_FILE" ]; then
    HF_TOKEN=$(<"$TOKEN_FILE")
    HF_TOKEN=$(echo "$HF_TOKEN" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')

    if [ -n "$HF_TOKEN" ]; then
        echo "Hugging Face token found and read from $TOKEN_FILE."
        accelerate config default
        huggingface-cli login --token "$HF_TOKEN"
    else
        echo "Error: Hugging Face token file $TOKEN_FILE is empty."
        echo "Please ensure your token is in the file."
        exit 1
    fi
else
    echo "Error: Hugging Face token file $TOKEN_FILE not found."
    echo "Please create this file and put your Hugging Face token in it."
    exit 1
fi

accelerate config default
accelerate launch flux_inference_test.py
