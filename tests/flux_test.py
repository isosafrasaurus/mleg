from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    use_auth_token=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)

pipe.parallelize()
pipe.enable_attention_slicing()

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
img = pipe(prompt).images[0]
img.save("out.png")
print("Done! Saved to out.png")
