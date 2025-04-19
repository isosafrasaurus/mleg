import os
import math
import shutil
import logging
import itertools
import copy
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import TrainingConfig
from .data import DreamBoothDataset, collate_fn, PromptDataset
from .model_utils import (
    import_model_class_from_model_name_or_path,
    load_text_encoders,
    encode_prompt,
    unwrap_model,
    get_sigmas,
    log_validation,
    save_model_card,
)
from transformers import CLIPTokenizer, T5TokenizerFast
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
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from huggingface_hub import create_repo, upload_folder
from huggingface_hub.utils import insecure_hashlib

logger = get_logger(__name__)


def train(cfg: TrainingConfig) -> None:
    """
    Runs the full DreamBooth LoRA training loop using FluxPipeline.
    """
    # Check minimal diffusers version
    from diffusers.utils import check_min_version, is_wandb_available
    check_min_version("0.34.0.dev0")

    # Setup accelerator and logging
    output_dir = Path(cfg.output_dir)
    logging_dir = output_dir / cfg.logging_dir
    project_config = ProjectConfiguration(project_dir=str(output_dir), logging_dir=str(logging_dir))
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        mixed_precision=cfg.mixed_precision,
        log_with=cfg.report_to,
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    # Disable native AMP on MPS
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    # WandB check
    if cfg.report_to == "wandb" and accelerator.is_local_main_process:
        if not is_wandb_available():
            raise ImportError("Install wandb to use report_to=wandb.")

    # Logging configuration
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        import diffusers
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        import diffusers
        from transformers import logging as hf_logging
        hf_logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Seed
    if cfg.seed is not None:
        set_seed(cfg.seed)

    # ------------------------------------------------
    # Generate class images if using prior preservation
    # ------------------------------------------------
    if cfg.with_prior_preservation:
        class_dir = Path(cfg.class_data_dir)
        class_dir.mkdir(parents=True, exist_ok=True)
        existing = list(class_dir.iterdir())
        cur_count = len(existing)
        if cur_count < cfg.num_class_images:
            # choose dtype
            has_fp_accel = torch.cuda.is_available() or torch.backends.mps.is_available()
            torch_dtype = torch.bfloat16 if cfg.prior_generation_precision == "bf16" else (
                torch.float16 if has_fp_accel else torch.float32
            )
            if cfg.prior_generation_precision == "fp32":
                torch_dtype = torch.float32
            elif cfg.prior_generation_precision == "fp16":
                torch_dtype = torch.float16

            pipeline = FluxPipeline.from_pretrained(
                cfg.pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                revision=cfg.revision,
                variant=cfg.variant,
            )
            pipeline.set_progress_bar_config(disable=True)

            n_to_gen = cfg.num_class_images - cur_count
            sample_dataset = PromptDataset(cfg.class_prompt, n_to_gen)
            sample_loader = DataLoader(sample_dataset, batch_size=cfg.sample_batch_size)
            sample_loader = accelerator.prepare(sample_loader)
            pipeline.to(accelerator.device)

            for batch in tqdm(sample_loader, desc="Generating class images", disable=not accelerator.is_local_main_process):
                images = pipeline(batch["prompt"]).images
                for i, img in enumerate(images):
                    hash_img = insecure_hashlib.sha1(img.tobytes()).hexdigest()
                    img_path = class_dir / f"{batch['index'][i] + cur_count}-{hash_img}.jpg"
                    img.save(img_path)

            del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---------------------
    # Hub repo preparation
    # ---------------------
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        if cfg.push_to_hub:
            repo_id = create_repo(
                repo_id=cfg.hub_model_id or output_dir.name,
                exist_ok=True,
            ).repo_id

    # ----------------------
    # Load tokenizers & encoders
    # ----------------------
    tokenizer_one = CLIPTokenizer.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=cfg.revision,
    )
    tokenizer_two = T5TokenizerFast.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=cfg.revision,
    )

    cls_one = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path,
        cfg.revision,
    )
    cls_two = import_model_class_from_model_name_or_path(
        cfg.pretrained_model_name_or_path,
        cfg.revision,
        subfolder="text_encoder_2",
    )

    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    text_encoder_one, text_encoder_two = load_text_encoders(
        cls_one, cls_two,
        cfg.pretrained_model_name_or_path,
        cfg.revision,
        cfg.variant,
    )
    vae = AutoencoderKL.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="vae",
        revision=cfg.revision,
        variant=cfg.variant,
    )
    transformer = FluxTransformer2DModel.from_pretrained(
        cfg.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=cfg.revision,
        variant=cfg.variant,
    )

    # ----------------
    # Freeze base nets
    # ----------------
    for m in (transformer, vae, text_encoder_one, text_encoder_two):
        m.requires_grad_(False)

    # determine weight dtype
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
        raise ValueError("Mixed precision bf16 is not supported on MPS.")

    # move to device
    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)

    # -------------------------------
    # Gradient checkpointing & LoRA
    # -------------------------------
    if cfg.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        if cfg.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()

    target_modules = cfg.lora_layers or [
        "attn.to_k","attn.to_q","attn.to_v","attn.to_out.0",
        "attn.add_k_proj","attn.add_q_proj","attn.add_v_proj","attn.to_add_out",
        "ff.net.0.proj","ff.net.2","ff_context.net.0.proj","ff_context.net.2",
    ]
    lora_config = LoraConfig(
        r=cfg.rank,
        lora_alpha=cfg.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(lora_config)
    if cfg.train_text_encoder:
        text_lora_config = LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.rank,
            init_lora_weights="gaussian",
            target_modules=["q_proj","k_proj","v_proj","out_proj"],
        )
        text_encoder_one.add_adapter(text_lora_config)

    # ----------------
    # Save/load hooks
    # ----------------
    def save_hook(models, weights, output_dir_path):
        if accelerator.is_main_process:
            transformer_lora = None
            text_lora = None
            for m in models:
                if isinstance(m, type(unwrap_model(transformer))):
                    transformer_lora = get_peft_model_state_dict(m)
                else:
                    text_lora = get_peft_model_state_dict(m)
                weights.pop()
            FluxPipeline.save_lora_weights(
                output_dir_path,
                transformer_lora_layers=transformer_lora,
                text_encoder_lora_layers=text_lora,
            )

    def load_hook(models, input_dir_path):
        transformer_model = None
        text_model = None
        while models:
            m = models.pop()
            if isinstance(m, type(unwrap_model(transformer))): transformer_model = m
            else: text_model = m
        lora_state = FluxPipeline.lora_state_dict(input_dir_path)
        # transformer keys
        tr_state = {k.replace("transformer.", ""): v for k, v in lora_state.items() if k.startswith("transformer.")}
        tr_state = convert_unet_state_dict_to_peft(tr_state)
        set_peft_model_state_dict(transformer_model, tr_state)
        if cfg.train_text_encoder:
            _set_state_dict_into_text_encoder(lora_state, prefix="text_encoder.", text_encoder=text_model)
        if accelerator.mixed_precision == "fp16":
            cast_training_params([transformer_model] + ([text_model] if cfg.train_text_encoder else []))

    accelerator.register_save_state_pre_hook(save_hook)
    accelerator.register_load_state_pre_hook(load_hook)

    # ----------------
    # Scale learning rate
    # ----------------
    if cfg.scale_lr:
        cfg.learning_rate = (
            cfg.learning_rate
            * cfg.gradient_accumulation_steps
            * cfg.train_batch_size
            * accelerator.num_processes
        )

    # -------------------
    # Optimizer & scheduler
    # -------------------
    # prepare param groups
    transformer_params = [p for p in transformer.parameters() if p.requires_grad]
    param_groups = [{"params": transformer_params, "lr": cfg.learning_rate}]
    if cfg.train_text_encoder:
        text_params = [p for p in text_encoder_one.parameters() if p.requires_grad]
        param_groups.append({
            "params": text_params,
            "weight_decay": cfg.adam_weight_decay_text_encoder,
            "lr": cfg.text_encoder_lr,
        })

    # select optimizer
    if cfg.optimizer.lower() == "adamw":
        if cfg.use_8bit_adam:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                param_groups,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                weight_decay=cfg.adam_weight_decay,
                eps=cfg.adam_epsilon,
            )
        else:
            optimizer = torch.optim.AdamW(
                param_groups,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                weight_decay=cfg.adam_weight_decay,
                eps=cfg.adam_epsilon,
            )
    elif cfg.optimizer.lower() == "prodigy":
        import prodigyopt
        optimizer = prodigyopt.Prodigy(
            param_groups,
            betas=(cfg.adam_beta1, cfg.adam_beta2),
            beta3=cfg.prodigy_beta3,
            weight_decay=cfg.adam_weight_decay,
            eps=cfg.adam_epsilon,
            decouple=cfg.prodigy_decouple,
            use_bias_correction=cfg.prodigy_use_bias_correction,
            safeguard_warmup=cfg.prodigy_safeguard_warmup,
        )
    else:
        logging.warning("Unsupported optimizer type; defaulting to AdamW.")
        optimizer = torch.optim.AdamW(param_groups, lr=cfg.learning_rate)

    # create scheduler
    overrode_steps = False
    if cfg.max_train_steps is None:
        # will compute after dataloader init
        pass

    # -------------------
    # DataLoader
    # -------------------
    train_dataset = DreamBoothDataset(
        instance_data_dir=Path(cfg.instance_data_dir) if cfg.instance_data_dir else None,
        instance_prompt=cfg.instance_prompt,
        dataset_name=cfg.dataset_name,
        dataset_config_name=cfg.dataset_config_name,
        cache_dir=Path(cfg.cache_dir) if cfg.cache_dir else None,
        image_column=cfg.image_column,
        caption_column=cfg.caption_column,
        repeats=cfg.repeats,
        class_prompt=cfg.class_prompt,
        class_data_root=Path(cfg.class_data_dir) if cfg.with_prior_preservation else None,
        class_num=cfg.num_class_images,
        size=cfg.resolution,
        resolution=cfg.resolution,
        center_crop=cfg.center_crop,
        random_flip=cfg.random_flip,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_fn(examples, cfg.with_prior_preservation),
        num_workers=cfg.dataloader_num_workers,
    )

    # compute total steps / create scheduler
    num_update_steps = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    if cfg.max_train_steps is None:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps
        overrode_steps = True
    lr_scheduler = get_scheduler(
        cfg.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=cfg.max_train_steps * accelerator.num_processes,
        num_cycles=cfg.lr_num_cycles,
        power=cfg.lr_power,
    )

    # ----------------
    # Prepare with accelerator
    # ----------------
    if cfg.train_text_encoder:
        transformer, text_encoder_one, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            transformer, text_encoder_one, optimizer, train_loader, lr_scheduler
        )
    else:
        transformer, optimizer, train_loader, lr_scheduler = accelerator.prepare(
            transformer, optimizer, train_loader, lr_scheduler
        )

    # re-calc if needed
    num_update_steps = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    if overrode_steps:
        cfg.max_train_steps = cfg.num_train_epochs * num_update_steps
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps)

    # ----------------
    # Initialize trackers
    # ----------------
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth-flux-dev-lora", config=vars(cfg))

    # ----------------
    # Training loop
    # ----------------
    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    logger.info(f"***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Number of optimization steps = {cfg.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # resume from checkpoint
    if cfg.resume_from_checkpoint:
        # same logic as original, selecting latest or specified, then accelerator.load_state
        pass

    progress_bar = tqdm(
        range(cfg.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    for epoch in range(first_epoch, cfg.num_train_epochs):
        transformer.train()
        if cfg.train_text_encoder:
            text_encoder_one.train()
            unwrap_model(text_encoder_one).text_model.embeddings.requires_grad_(True)

        for step, batch in enumerate(train_loader):
            models_to_accumulate = [transformer]
            if cfg.train_text_encoder:
                models_to_accumulate.append(text_encoder_one)
            with accelerator.accumulate(models_to_accumulate):
                # prepare text embeddings
                prompts = batch["prompts"]
                if train_dataset.custom_instance_prompts:
                    if not cfg.train_text_encoder:
                        prompt_embeds, pooled_embeds, text_ids = encode_prompt(
                            [text_encoder_one, text_encoder_two],
                            [None, None],
                            prompts,
                            cfg.max_sequence_length,
                            device=accelerator.device,
                        )
                    else:
                        # full encode for each prompt
                        prompt_embeds, pooled_embeds, text_ids = encode_prompt(
                            [text_encoder_one, text_encoder_two],
                            [tokenizer_one, tokenizer_two],
                            prompts,
                            cfg.max_sequence_length,
                            device=accelerator.device,
                        )
                else:
                    if cfg.train_text_encoder:
                        tokens1 = tokenizer_one(
                            cfg.instance_prompt,
                            padding="max_length",
                            max_length=77,
                            return_tensors="pt",
                        ).input_ids.repeat(len(prompts), 1)
                        tokens2 = tokenizer_two(
                            cfg.instance_prompt,
                            padding="max_length",
                            max_length=cfg.max_sequence_length,
                            return_tensors="pt",
                        ).input_ids.repeat(len(prompts), 1)
                        prompt_embeds, pooled_embeds, text_ids = encode_prompt(
                            [text_encoder_one, text_encoder_two],
                            [None, None],
                            cfg.instance_prompt,
                            cfg.max_sequence_length,
                            device=accelerator.device,
                            text_input_ids_list=[tokens1, tokens2],
                        )
                    else:
                        prompt_embeds = encoder_hidden_states
                        pooled_embeds = pooled_hidden_states
                        text_ids = None

                # encode latents
                pixel_vals = batch["pixel_values"].to(dtype=vae.dtype)
                latents = vae.encode(pixel_vals).latent_dist.sample()
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)

                # sample noise and timesteps
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=cfg.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=cfg.logit_mean,
                    logit_std=cfg.logit_std,
                    mode_scale=cfg.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=latents.device)

                sigmas = get_sigmas(timesteps, scheduler=noise_scheduler_copy, device=accelerator.device, dtype=latents.dtype)
                noisy_input = (1.0 - sigmas) * latents + sigmas * noise

                model_input = FluxPipeline._pack_latents(
                    noisy_input,
                    batch_size=bsz,
                    num_channels_latents=noisy_input.shape[1],
                    height=noisy_input.shape[2],
                    width=noisy_input.shape[3],
                )

                # guidance
                if unwrap_model(transformer).config.guidance_embeds:
                    guidance = torch.full((bsz,), cfg.guidance_scale, device=accelerator.device)
                else:
                    guidance = None

                # predict
                pred = transformer(
                    hidden_states=model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=FluxPipeline._prepare_latent_image_ids(
                        bsz,
                        latents.shape[2] // 2,
                        latents.shape[3] // 2,
                        accelerator.device,
                        weight_dtype,
                    ),
                )[0]
                pred = FluxPipeline._unpack_latents(
                    pred,
                    height=latents.shape[2] * (2 ** (len(vae.config.block_out_channels) - 1)),
                    width=latents.shape[3] * (2 ** (len(vae.config.block_out_channels) - 1)),
                    vae_scale_factor=2 ** (len(vae.config.block_out_channels) - 1),
                )

                # loss
                weight = compute_loss_weighting_for_sd3(weighting_scheme=cfg.weighting_scheme, sigmas=sigmas)
                target = noise - latents
                if cfg.with_prior_preservation:
                    pred, pred_prior = torch.chunk(pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)
                    prior_loss = torch.mean((weight * (pred_prior - target_prior) ** 2).reshape(target_prior.shape[0], -1), 1).mean()
                loss = torch.mean((weight * (pred - target) ** 2).reshape(target.shape[0], -1), 1).mean()
                if cfg.with_prior_preservation:
                    loss = loss + cfg.prior_loss_weight * prior_loss

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, cfg.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                if accelerator.is_main_process and global_step % cfg.checkpointing_steps == 0:
                    # manage old checkpoints
                    if cfg.checkpoints_total_limit is not None:
                        all_ckpts = sorted(
                            p.name for p in output_dir.iterdir() if p.name.startswith("checkpoint")
                        )
                        if len(all_ckpts) >= cfg.checkpoints_total_limit:
                            for rm in all_ckpts[: len(all_ckpts) - cfg.checkpoints_total_limit + 1]:
                                shutil.rmtree(output_dir / rm)
                    save_path = output_dir / f"checkpoint-{global_step}"
                    accelerator.save_state(str(save_path))
                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step >= cfg.max_train_steps:
                break

        # validation
        if cfg.validation_prompt and epoch % cfg.validation_epochs == 0 and accelerator.is_main_process:
            pipeline = FluxPipeline.from_pretrained(
                cfg.pretrained_model_name_or_path,
                vae=vae,
                text_encoder=unwrap_model(text_encoder_one),
                text_encoder_2=unwrap_model(text_encoder_two),
                transformer=unwrap_model(transformer),
                revision=cfg.revision,
                variant=cfg.variant,
                torch_dtype=weight_dtype,
            )
            images = log_validation(
                pipeline,
                cfg.validation_prompt,
                cfg.num_validation_images,
                cfg.seed,
                accelerator,
                epoch,
                weight_dtype,
            )
            del pipeline

    # Final save & push
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_tr = unwrap_model(transformer)
        if cfg.upcast_before_saving:
            final_tr.to(torch.float32)
        FluxPipeline.save_lora_weights(
            save_directory=str(output_dir),
            transformer_lora_layers=get_peft_model_state_dict(final_tr),
            text_encoder_lora_layers=None,
        )
        if cfg.push_to_hub:
            # final inference
            pipeline = FluxPipeline.from_pretrained(
                cfg.pretrained_model_name_or_path,
                revision=cfg.revision,
                variant=cfg.variant,
                torch_dtype=weight_dtype,
            )
            pipeline.load_lora_weights(str(output_dir))
            images = []
            if cfg.validation_prompt and cfg.num_validation_images > 0:
                images = log_validation(
                    pipeline,
                    cfg.validation_prompt,
                    cfg.num_validation_images,
                    cfg.seed,
                    accelerator,
                    epoch,
                    weight_dtype,
                    is_final_validation=True,
                )
            save_model_card(
                repo_id,
                images=images,
                base_model=cfg.pretrained_model_name_or_path,
                train_text_encoder=cfg.train_text_encoder,
                instance_prompt=cfg.instance_prompt,
                validation_prompt=cfg.validation_prompt,
                repo_folder=str(output_dir),
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=str(output_dir),
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )
            del pipeline

    accelerator.end_training()
