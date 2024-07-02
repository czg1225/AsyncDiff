from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, AutoencoderKL
from diffusers.utils import load_image, make_image_grid
from PIL import Image
import cv2
import numpy as np
import torch
import torch.distributed as dist
from asyncdiff.async_sd import AsyncDiff
import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-xl-base-1.0')     #model= 'runwayml/stable-diffusion-v1-5'
    parser.add_argument("--prompt", type=str, default="aerial view, a futuristic research complex in a bright foggy jungle, hard lighting")
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--model_n", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--warm_up", type=int, default=3)
    parser.add_argument("--time_shift", type=bool, default=False)
    args = parser.parse_args()

    original_image = load_image(
        "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd_controlnet/hf-logo.png"
    )

    image = np.array(original_image)

    low_threshold = 100
    high_threshold = 200

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    make_image_grid([original_image, canny_image], rows=1, cols=2)

    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)
    pipeline = StableDiffusionXLControlNetPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        low_cpu_mem_usage=True
    )
 

    async_diff = AsyncDiff(pipeline, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)

    negative_prompt = 'low quality, bad quality, sketches'

    # warm up
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    image = pipeline(
        args.prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        controlnet_conditioning_scale=0.5,
    ).images[0]

    #Inference
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    start = time.time()
    image = pipeline(
        args.prompt,
        negative_prompt=negative_prompt,
        image=canny_image,
        controlnet_conditioning_scale=0.5,
    ).images[0]
    print(f"Rank {dist.get_rank()} Time taken: {time.time()-start:.2f} seconds.")

    if dist.get_rank() == 0:
        output = make_image_grid([original_image, canny_image, image], rows=1, cols=3)
        output.save(f"output.jpg")
