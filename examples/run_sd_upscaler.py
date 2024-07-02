import torch
import torch.distributed as dist
from asyncdiff.async_sd import AsyncDiff
import time
import argparse
import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='stabilityai/stable-diffusion-x4-upscaler')     #model= 'runwayml/stable-diffusion-v1-5'
    parser.add_argument("--prompt", type=str, default='a white cat')
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--model_n", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--warm_up", type=int, default=1)
    parser.add_argument("--time_shift", type=bool, default=False)
    args = parser.parse_args()

    pipeline = StableDiffusionUpscalePipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, 
        use_safetensors=True, low_cpu_mem_usage=True
    )

    url = "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/sd2-upscale/low_res_cat.png"
    response = requests.get(url)
    low_res_img = Image.open(BytesIO(response.content)).convert("RGB")
    low_res_img = low_res_img.resize((128, 128))

    async_diff = AsyncDiff(pipeline, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)

    # warm up   
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    image = pipeline(
        prompt=args.prompt,
        image=low_res_img,
    ).images[0]


    # inference
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    start = time.time()
    image = pipeline(
        prompt=args.prompt,
        image=low_res_img,
    ).images[0]
    print(f"Rank {dist.get_rank()} Time taken: {time.time()-start:.2f} seconds.")

    image.save(f"output.jpg")

    if dist.get_rank() == 0:
        image.save(f"output.jpg")