from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch
import torch.distributed as dist
from asyncdiff.async_sd import AsyncDiff
import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='diffusers/stable-diffusion-xl-1.0-inpainting-0.1')    
    parser.add_argument("--prompt", type=str, default="a tiger sitting on a park bench")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--model_n", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--warm_up", type=int, default=1)
    parser.add_argument("--time_shift", type=bool, default=False)
    args = parser.parse_args()

    pipeline = AutoPipelineForInpainting.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16")

    async_diff = AsyncDiff(pipeline, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)

    img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
    mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

    image = load_image(img_url).resize((1024, 1024))
    mask_image = load_image(mask_url).resize((1024, 1024))

    prompt = args.prompt
    generator = torch.Generator(device="cuda").manual_seed(0)

    # warm up
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    image = pipeline(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    guidance_scale=8.0,
    num_inference_steps=30,  # steps between 15 and 30 work well for us
    strength=0.99,  # make sure to use `strength` below 1.0
    ).images[0]

    # inference
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    start = time.time()
    image = pipeline(
    prompt=prompt,
    image=image,
    mask_image=mask_image,
    guidance_scale=8.0,
    num_inference_steps=30,  # steps between 15 and 30 work well for us
    strength=0.99,  # make sure to use `strength` below 1.0
    ).images[0]
    print(f"Rank {dist.get_rank()} Time taken: {time.time()-start:.2f} seconds.")

    if dist.get_rank() == 0:
        image.save(f"output.jpg")