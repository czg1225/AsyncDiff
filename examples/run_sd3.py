import torch
from diffusers import StableDiffusion3Pipeline
import torch.distributed as dist
from asyncdiff.async_sd3 import AsyncDiff
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-3-medium-diffusers')     
    parser.add_argument("--prompt", type=str, default='A cat holding a sign that says hello world')
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--model_n", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--warm_up", type=int, default=1)
    parser.add_argument("--time_shift", type=bool, default=False)
    args = parser.parse_args()

    pipeline = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    async_diff = AsyncDiff(pipeline, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)

    # warm up
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    image = pipeline(
        args.prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]

    # inference
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    start = time.time()
    image = pipeline(
        args.prompt,
        negative_prompt="",
        num_inference_steps=28,
        guidance_scale=7.0,
    ).images[0]
    print(f"Time taken: {time.time()-start:.2f} seconds.")

    if dist.get_rank() == 0:
        image.save(f"output.jpg")
