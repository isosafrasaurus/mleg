# model_utils.py

import os
import copy
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import PretrainedConfig, CLIPTextModel, T5EncoderModel
from transformers import CLIPTokenizer, T5TokenizerFast
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxPipeline,
    FluxTransformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    _set_state_dict_into_text_encoder,
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import (
    check_min_version,
    convert_unet_state_dict_to_peft,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module
from PIL import Image
from tqdm.auto import tqdm


__all__ = [
    "import_model_class_from_model_name_or_path",
    "load_text_encoders",
    "_encode_prompt_with_t5",
    "_encode_prompt_with_clip",
    "encode_prompt",
    "log_validation",
    "save_model_card",
    "unwrap_model",
    "get_sigmas",
]


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str,
    revision: Optional[str] = None,
    subfolder: str = "text_encoder",
) -> torch.nn.Module:
    """
    Inspect the pretrained_config.architectures to choose the correct text encoder class.

    Returns CLIPTextModel or T5EncoderModel.
    """
    config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder=subfolder,
        revision=revision,
    )
    architecture = config.architectures[0]
    if architecture == "CLIPTextModel":
        return CLIPTextModel
    elif architecture == "T5EncoderModel":
        return T5EncoderModel
    else:
        raise ValueError(f"Unsupported text encoder architecture: {architecture}")


def load_text_encoders(
    text_encoder_cls_one: type,
    text_encoder_cls_two: type,
    pretrained_model_name_or_path: str,
    revision: Optional[str] = None,
    variant: Optional[str] = None,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Load two text encoders from the same pretrained model path.
    Assumes subfolders 'text_encoder' and 'text_encoder_2'.
    """
    encoder_one = text_encoder_cls_one.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
        variant=variant,
    )
    encoder_two = text_encoder_cls_two.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=revision,
        variant=variant,
    )
    return encoder_one, encoder_two


def _encode_prompt_with_t5(
    text_encoder: torch.nn.Module,
    tokenizer: Optional[T5TokenizerFast],
    max_sequence_length: int,
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    text_input_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Encode prompts with a T5-based text encoder.
    Returns prompt hidden states of shape (batch * num_images_per_prompt, seq_len, dim).
    """
    prompts = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompts)

    if tokenizer is not None:
        inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("text_input_ids must be provided if tokenizer is None")

    device = device or next(text_encoder.parameters()).device
    embeds = text_encoder(text_input_ids.to(device))[0]
    dtype = getattr(text_encoder, "dtype", embeds.dtype)
    embeds = embeds.to(device=device, dtype=dtype)

    _, seq_len, _ = embeds.shape
    # repeat for multiple images per prompt
    embeds = embeds.repeat(1, num_images_per_prompt, 1)
    embeds = embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    return embeds


def _encode_prompt_with_clip(
    text_encoder: torch.nn.Module,
    tokenizer: Optional[CLIPTokenizer],
    prompt: Union[str, List[str]],
    num_images_per_prompt: int = 1,
    device: Optional[torch.device] = None,
    text_input_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Encode prompts with a CLIP-based text encoder.
    Returns pooled output repeated per image.
    """
    prompts = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompts)

    if tokenizer is not None:
        inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = inputs.input_ids
    elif text_input_ids is None:
        raise ValueError("text_input_ids must be provided if tokenizer is None")

    device = device or next(text_encoder.parameters()).device
    outputs = text_encoder(text_input_ids.to(device), output_hidden_states=False)
    pooled = outputs.pooler_output

    dtype = getattr(text_encoder, "dtype", pooled.dtype)
    pooled = pooled.to(device=device, dtype=dtype)

    # repeat for multiple images per prompt
    pooled = pooled.repeat(1, num_images_per_prompt, 1)
    pooled = pooled.view(batch_size * num_images_per_prompt, -1)
    return pooled


def encode_prompt(
    text_encoders: List[torch.nn.Module],
    tokenizers: List[Optional[Union[CLIPTokenizer, T5TokenizerFast]]],
    prompt: Union[str, List[str]],
    max_sequence_length: int,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    text_input_ids_list: Optional[List[torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Combined encoding for a two‑encoder setup (CLIP + T5).
    Returns (encoder_hidden_states, pooled_hidden_states, zero text_ids).
    """
    # CLIP part
    pooled = _encode_prompt_with_clip(
        text_encoder=text_encoders[0],
        tokenizer=tokenizers[0],
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=(text_input_ids_list[0] if text_input_ids_list else None),
    )

    # T5 part
    embeds = _encode_prompt_with_t5(
        text_encoder=text_encoders[1],
        tokenizer=tokenizers[1],
        max_sequence_length=max_sequence_length,
        prompt=prompt,
        num_images_per_prompt=num_images_per_prompt,
        device=device,
        text_input_ids=(text_input_ids_list[1] if text_input_ids_list else None),
    )

    # dummy text_ids for FluxPipeline compatibility
    batch_size = (len(prompt) if isinstance(prompt, list) else 1)
    seq_len = embeds.shape[1]
    dtype = embeds.dtype
    text_ids = torch.zeros(batch_size * num_images_per_prompt, 3, device=device, dtype=dtype)

    return embeds, pooled, text_ids


def log_validation(
    pipeline: FluxPipeline,
    validation_prompt: str,
    num_validation_images: int,
    seed: Optional[int],
    accelerator,
    epoch: int,
    torch_dtype: torch.dtype,
    is_final_validation: bool = False,
) -> List[Image.Image]:
    """
    Run validation to generate images and log to TensorBoard/W&B trackers.
    Returns the list of PIL Images.
    """
    device = accelerator.device
    # move pipeline and disable its progress bar
    pipeline = pipeline.to(device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(seed) if seed is not None else None
    autocast_ctx = torch.autocast(device.type) if not is_final_validation else torch.cpu.amp.autocast(False)

    # precompute prompt embeddings
    with torch.no_grad():
        prompt_embeds, pooled_embeds, text_ids = pipeline.encode_prompt(
            validation_prompt, prompt_2=validation_prompt
        )

    images: List[Image.Image] = []
    for _ in range(num_validation_images):
        with autocast_ctx:
            img = pipeline(
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_embeds,
                generator=generator,
            ).images[0]
            images.append(img)

    # log to trackers
    for tracker in accelerator.trackers:
        name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            arr = np.stack([np.asarray(i) for i in images])
            tracker.writer.add_images(name, arr, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log({name: [tracker.Image(img, caption=validation_prompt) for img in images]})

    del pipeline
    free_memory()
    return images


def save_model_card(
    repo_id: str,
    repo_folder: Union[str, Path],
    images: Optional[List[Image.Image]] = None,
    base_model: Optional[str] = None,
    train_text_encoder: bool = False,
    instance_prompt: Optional[str] = None,
    validation_prompt: Optional[str] = None,
) -> None:
    """
    Save a model card with an optional image gallery to a local folder and HF Hub.
    """
    repo_folder = str(repo_folder)
    widget: List[dict] = []
    if images:
        for idx, img in enumerate(images):
            path = os.path.join(repo_folder, f"image_{idx}.png")
            img.save(path)
            widget.append({"text": validation_prompt or "", "output": {"url": f"image_{idx}.png"}})

    description = f"""
# Flux DreamBooth LoRA - {repo_id}

<Gallery />

These are {repo_id} DreamBooth LoRA weights trained on {base_model}.
LoRA on text encoder: {train_text_encoder}.

Trigger word: `{instance_prompt}`.
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        prompt=instance_prompt,
        model_description=description,
        widget=widget,
    )
    tags = [
        "text-to-image",
        "diffusers-training",
        "lora",
        "flux-diffusers",
    ]
    model_card = populate_model_card(model_card, tags=tags)
    model_card.save(os.path.join(repo_folder, "README.md"))


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """
    Unwrap a potentially compiled or accelerator-wrapped model.
    """
    from diffusers.utils.torch_utils import is_compiled_module
    if hasattr(model, "module"):
        core = model.module
    else:
        core = model
    return core._orig_mod if is_compiled_module(core) else core


def get_sigmas(
    timesteps: torch.Tensor,
    scheduler: FlowMatchEulerDiscreteScheduler,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
    n_dim: int = 4,
) -> torch.Tensor:
    """
    Given integer timesteps, return the matching sigma values from scheduler.sigmas,
    reshaped to have at least `n_dim` dims for broadcasting.
    """
    device = device or timesteps.device
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_ts = scheduler.timesteps.to(device=device)
    # find index for each timestep value
    indices = [(schedule_ts == t).nonzero().item() for t in timesteps]
    out = sigmas[indices]
    out = out.view(-1)
    while out.ndim < n_dim:
        out = out.unsqueeze(-1)
    return out
