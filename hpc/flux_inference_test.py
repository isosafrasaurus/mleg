import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16,       # or torch.float16 if you prefer
    safety_checker=None,               # (optional) if you want to disable safety checks
    device_map="balanced"
)
pipe = pipe.to("cuda")               # move to GPU
pipe.enable_model_cpu_offload()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipe(prompt).images[0]
image.save("astronaut_in_jungle.png")

