#!/usr/bin/env python
import torch
from diffusers import UNet2DModel, DDPMScheduler

def main():
    # 1) Basic CUDA sanity check
    print("CUDA available:", torch.cuda.is_available(), 
          "device count:", torch.cuda.device_count())

    # 2) Build a minimal UNet
    model = UNet2DModel(
        sample_size=64,            # 64×64 feature maps
        in_channels=3,             # RGB
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(64, 128),
        down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",),
    )
    model.to("cuda")

    # 3) Create a tiny DDPM scheduler
    scheduler = DDPMScheduler(num_train_timesteps=5)

    # 4) Random noise tensor
    noise = torch.randn(1, 3, 64, 64, device="cuda")

    # 5) One pass of the reverse diffusion loop
    with torch.no_grad():
        for t in scheduler.timesteps:
            output = model(noise, t)
            noise = output.sample  # next-noise

    print("Dummy UNet inference OK → output shape:", noise.shape)

if __name__ == "__main__":
    main()

