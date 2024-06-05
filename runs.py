import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from diffusers import StableDiffusionPipeline
from src.async_diff import AsyncDiff


pipeline = StableDiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-1", torch_dtype=torch.float16, 
    use_safetensors=True, low_cpu_mem_usage=False
)
pipeline = pipeline.to('cuda')
# async_diff = AsyncDiff(pipeline, model_n=2, stride=1)
async_diff = AsyncDiff(pipeline, model_n=3, stride=1)

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

pipeline(
    "(fractal crystal skin:1.1) with( ice crown:1.4) woman, white crystal skin, (fantasy:1.3), (Anna Dittmann:1.3)",
    num_inference_steps=50,
).images[0].save(f"output_{dist.get_rank()}.jpg")
# ).images[0].save(f"output.jpg")
