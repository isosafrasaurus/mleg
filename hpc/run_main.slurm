#!/bin/bash
#SBATCH --job-name=flux8G_multinode
#SBATCH --account=commons
#SBATCH --partition=long
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=/home/pzz1/logs/%j.log

# Load a clean environment and needed modules
module purge
module load GCC/12.3.0 OpenMPI/4.1.5 PyTorch/2.1.2-CUDA-12.1.1

if [[ -n "$SLURM_JOB_ID" ]]; then
  JOB_ID="$SLURM_JOB_ID"
else
  JOB_ID="$(date +%s)-$$"
fi

# Prepare a per-job virtualenv on fast shared scratch
SCRATCH_ENV="$SHARED_SCRATCH/$USER/gpuenv_$JOB_ID"
mkdir -p   "$SCRATCH_ENV"
python3 -m venv "$SCRATCH_ENV"
source    "$SCRATCH_ENV/bin/activate"

# Redirect pip cache to your scratch
export PIP_CACHE_DIR="$SHARED_SCRATCH/$USER/.cache/pip"
mkdir -p "$PIP_CACHE_DIR"

# Copy your inference script into scratch
cp "$HOME/mleg/hpc/flux_inference_test.py" "$SCRATCH_ENV/"

# Install Python dependencies
pip install bitsandbytes transformers accelerate peft
# Install your local diffusers build
cp -r "$HOME/git_to_compute/diffusers" "$SCRATCH_ENV/"
pip install "$SCRATCH_ENV/diffusers"
pip install sentencepiece

# Configure Hugging Face caches off your home dir
export HF_HOME="$SHARED_SCRATCH/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export DIFFUSERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Non-interactive HF login via token file
TOKEN_FILE="$HOME/.HF_TOKEN"
if [[ -f "$TOKEN_FILE" ]]; then
  HF_TOKEN=$(<"$TOKEN_FILE")
  HF_TOKEN="${HF_TOKEN//[[:space:]]/}"
  if [[ -n "$HF_TOKEN" ]]; then
    echo "Found HF token, logging in..."
    huggingface-cli login --token "$HF_TOKEN"
  else
    echo "ERROR: $TOKEN_FILE is empty." >&2
    exit 1
  fi
else
  echo "ERROR: HF token file not found at $TOKEN_FILE." >&2
  exit 1
fi

accelerate config default
accelerate launch flux_inference_test.py
