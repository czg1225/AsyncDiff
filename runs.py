import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import StableDiffusionPipeline
from src.async_diff import AsyncDiff
import time


pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, 
    use_safetensors=True, low_cpu_mem_usage=False
)

async_diff = AsyncDiff(pipeline, model_n=3, stride=2, time_shift=False)

# torch.manual_seed(20)
# torch.cuda.manual_seed_all(20)
# async_diff.reset_state(warm_up=1)
# image = pipeline(
#     "Impressionist style, a yellow rubber duck floating on the wave on the sunset",
#     num_inference_steps=50,
# ).images[0]

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
async_diff.reset_state(warm_up=9)
start = time.time()
image = pipeline(
    "A kitten that is sitting down by a door",
    num_inference_steps=50,
).images[0]
print(f"Time taken: {time.time()-start:.2f} seconds.")

image.save(f"output_{dist.get_rank()}.jpg")


