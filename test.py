from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",  # ← 이걸로
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

prompt = "GPU network visualization, digital art, blue neon"
image = pipe(prompt).images[0]

image.save("output2.png")
print("output.png 저장 완료!")