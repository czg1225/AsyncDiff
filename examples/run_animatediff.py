import torch
import torch.distributed as dist
from diffusers.utils import export_to_gif
from asyncdiff.async_animate import AsyncDiff
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
import time
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='emilianJR/epiCRealism')    
    parser.add_argument("--prompt", type=str, default='Brilliant fireworks, high quality')
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--model_n", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--warm_up", type=int, default=2)
    parser.add_argument("--time_shift", type=bool, default=False)
    args = parser.parse_args()

    adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
    pipeline = AnimateDiffPipeline.from_pretrained(args.model, motion_adapter=adapter, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    scheduler = DDIMScheduler.from_pretrained(
        args.model,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipeline.scheduler = scheduler

    async_diff = AsyncDiff(pipeline, model_n=args.model_n, stride=args.stride, time_shift=args.time_shift)

    # warm up
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    frames = pipeline(
            prompt=args.prompt,
            negative_prompt="bad quality, worse quality, low resolution",
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=50,
        ).frames[0]

    # inference
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    async_diff.reset_state(warm_up=args.warm_up)
    start = time.time()
    frames = pipeline(
            prompt=args.prompt,
            negative_prompt="bad quality, worse quality, low resolution",
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=50,
        ).frames[0]
    print(f"Rank {dist.get_rank()} Time taken: {time.time()-start:.2f} seconds.")

    if dist.get_rank() == 0:
        export_to_gif(frames, "animation_async.gif")