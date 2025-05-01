Large-scale text-to-image diffusion models offer unprecedented generative capabilities,
yet tailoring them to synthesize specific subjects, such as individuals, remains
a challenge. This project focuses on personalizing the Black Forest Labs
FLUX.1-dev model using Low-Rank Adaptation (LoRA). We describe
a preprocessing pipeline leveraging MTCNN and facial embeddings for robust subject
identification and data preparation. We then rigorously detail the mathematical
formulation of LoRA and its application to the attention and feed-forward layers
within FLUX.1-dev's core transformer model. We present training dynamics and
evaluation results, using placeholders based on observed trends, comparing different
LoRA ranks against baseline personalization techniques like Dreambooth.

Our findings indicate that LoRA provides an efficient and effective mechanism
for subject personalization within the FLUX.1-dev architecture, achieving
competitive identity preservation and prompt fidelity with
significantly fewer trainable parameters compared to full fine-tuning.
