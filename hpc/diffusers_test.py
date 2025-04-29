#!/usr/bin/env python
import torch
from diffusers import UNet2DModel, DDPMScheduler

def main():
    print("CUDA available:", torch.cuda.is_available(), 
          "device count:", torch.cuda.device_count())

    model = UNet2DModel(
        sample_size=64,               # 64×64 feature maps
        in_channels=3,                # RGB input
        out_channels=3,               # RGB output
        layers_per_block=1,
        block_out_channels=(64,),      # one entry one block
        down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",),
    )
    model.to("cuda")

    scheduler = DDPMScheduler(num_train_timesteps=5)

    noise = torch.randn(1, 3, 64, 64, device="cuda")

    with torch.no_grad():
        for t in scheduler.timesteps:
            out = model(noise, t)
            noise = out.sample

    print("Dummy UNet inference OK --> output shape:", noise.shape)

if __name__ == "__main__":
    main()

