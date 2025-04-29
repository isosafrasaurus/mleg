from diffusers import DiffusionPipeline
import torch, os
from huggingface_hub import snapshot_download

repo = snapshot_download("black-forest-labs/FLUX.1-dev", token=os.environ["HF_TOKEN"])
pipe = DiffusionPipeline.from_pretrained(
    repo,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

pipe.enable_attention_slicing()
pipe.to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
img = pipe(prompt).images[0]
img.save("out.png")
print("Done! Saved to out.png")
