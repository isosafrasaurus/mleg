# config.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


@dataclass
class TrainingConfig:
    pretrained_model_name_or_path: str
    instance_prompt: str

    # model & versioning
    revision: Optional[str] = None
    variant: Optional[str] = None

    # data sources
    dataset_name: Optional[str] = None
    dataset_config_name: Optional[str] = None
    instance_data_dir: Optional[Path] = None
    class_data_dir: Optional[Path] = None
    cache_dir: Optional[Path] = None

    # dataset columns & repeats
    image_column: str = "image"
    caption_column: Optional[str] = None
    repeats: int = 1

    # output & Hugging Face Hub
    output_dir: Path = Path("flux-dreambooth-lora")
    logging_dir: Path = Path("logs")
    push_to_hub: bool = False
    hub_token: Optional[str] = None
    hub_model_id: Optional[str] = None

    # image resolution & augmentations
    resolution: int = 512
    center_crop: bool = False
    random_flip: bool = False
    cache_latents: bool = False

    # validation
    validation_prompt: Optional[str] = None
    num_validation_images: int = 4
    validation_epochs: int = 50

    # prior preservation
    with_prior_preservation: bool = False
    prior_loss_weight: float = 1.0
    num_class_images: int = 100
    prior_generation_precision: Optional[str] = None  # "no","fp32","fp16","bf16"

    # optimization hyperparameters
    learning_rate: float = 1e-4
    text_encoder_lr: float = 5e-6
    guidance_scale: float = 3.5

    optimizer: str = "AdamW"        # or "prodigy"
    use_8bit_adam: bool = False

    # Adam / Prodigy specific
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-08
    adam_weight_decay: float = 1e-04
    adam_weight_decay_text_encoder: float = 1e-03

    prodigy_beta3: Optional[float] = None
    prodigy_decouple: bool = True
    prodigy_use_bias_correction: bool = True
    prodigy_safeguard_warmup: bool = True

    # LoRA settings
    rank: int = 4
    lora_layers: Optional[List[str]] = field(default=None)

    # learning-rate scheduler
    lr_scheduler: str = "constant"  # ["linear","cosine","cosine_with_restarts","polynomial","constant_with_warmup"]
    lr_warmup_steps: int = 500
    lr_num_cycles: int = 1
    lr_power: float = 1.0
    scale_lr: bool = False

    # batching & epochs
    train_batch_size: int = 4
    sample_batch_size: int = 4
    num_train_epochs: int = 1
    max_train_steps: Optional[int] = None

    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0

    dataloader_num_workers: int = 0
    local_rank: int = -1

    # timestep weighting for flow match
    weighting_scheme: str = "none"  # ["sigma_sqrt","logit_normal","mode","cosmap","none"]
    logit_mean: float = 0.0
    logit_std: float = 1.0
    mode_scale: float = 1.29

    # mixed precision & device flags
    mixed_precision: Optional[str] = None  # "no","fp16","bf16"
    allow_tf32: bool = False
    upcast_before_saving: bool = False

    # logging integrations & seed
    report_to: str = "tensorboard"  # or "wandb","comet_ml","all"
    seed: Optional[int] = None

    # checkpointing
    checkpointing_steps: int = 500
    checkpoints_total_limit: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
