#usr/bin/env bash
set -euo pipefail

export HF_HOME="$SHARED_SCRATCH/$USER/.cache/huggingface"
export DIFFUSERS_CACHE="$HF_HOME/diffusers"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$DIFFUSERS_CACHE" "$TRANSFORMERS_CACHE"

MODEL_ID="black-forest-labs/FLUX.1-dev"

echo "Caching FLUX.1-dev components for ${MODEL_ID}"
echo "→ HF_HOME = $HF_HOME"
echo

bin/python <<EOF
import os, torch

from diffusers import (
    FluxPipeline,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from transformers import (
    CLIPTokenizer,
    T5TokenizerFast,
    CLIPTextModelWithProjection,
    T5EncoderModel,
)

model_id = "${MODEL_ID}"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

print("Caching VAE")
AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=dtype, token=token)
print("Caching Scheduler")
FlowMatchEulerDiscreteScheduler.from_pretrained(
    model_id, subfolder="scheduler", torch_dtype=dtype, token=token
)
print("Caching Transformer")
FluxTransformer2DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=dtype, token=token
)
print("VAE, Scheduler & Transformer cached.\n")

print("Caching Tokenizers & Text Encoders")
CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", token=token)
T5TokenizerFast.from_pretrained(model_id, subfolder="tokenizer_2", token=token)
CLIPTextModelWithProjection.from_pretrained(
    model_id, subfolder="text_encoder", torch_dtype=dtype, token=token
)
T5EncoderModel.from_pretrained(
    model_id, subfolder="text_encoder_2", torch_dtype=dtype, token=token
)
print("Tokenizers & Text Encoders cached.\n")

FluxPipeline.from_pretrained(
    model_id,
    torch_dtype=dtype,
    token=token,
)
EOF
echo "Cache script complete."

