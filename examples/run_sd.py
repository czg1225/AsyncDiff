import torch
import torch.distributed as dist
from diffusers import StableDiffusionPipeline
from asyncdiff.async_sd import AsyncDiff
import time
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-2-1')     #model= 'runwayml/stable-diffusion-v1-5'
    parser.add_argument("--prompt", type=str, default='A kitten that is sitting down by a door')
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--model_n", type=int, default=3)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--warm_up", type=int, default=9)
    parser.add_argument("--time_shift", type=bool, default=False)
    args = parser.parse_args()

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, 
        use_safetensors=True, low_cpu_mem_usage=True
    )
    async_diff = AsyncDiff(pipeline, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)

    # warm up
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    image = pipeline(
        args.prompt,
        num_inference_steps=50,
    ).images[0]


    # inference
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    start = time.time()
    image = pipeline(
        args.prompt,
        num_inference_steps=50,
    ).images[0]
    print(f"Rank {dist.get_rank()} Time taken: {time.time()-start:.2f} seconds.")

    if dist.get_rank() == 0:
        image.save(f"output.jpg")