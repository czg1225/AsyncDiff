import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os
from diffusers.utils import load_image, export_to_video, export_to_gif



def run_inference(rank, world_size, config):
    print(f"Rank {rank} is running.")

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if config["strategy"] == "n2s1":
        from svd.n2s1 import AsyncDiffusionWorker
    elif config["strategy"] == "n3s1":
        from svd.n3s1 import AsyncDiffusionWorker
    elif config["strategy"] == "n4s1":
        from svd.n4s1 import AsyncDiffusionWorker


    # Load the conditioning image
    image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png?download=true")
    file_name = "rocket"
    image = image.resize((1024, 576))

    pipeline = AsyncDiffusionWorker(
        config,
    )
    torch.cuda.set_device(
        config["devices"][rank]
    )  # this is necessary to make sure the correct device is set
    

    for i in range(1):
        pipeline.reset_state(warm_up=config["warm_up"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        start = time.time()
        frames = pipeline(
            image, 
            decode_chunk_size=8,
            num_inference_steps=config["step"]
        ).frames[0]
        print(f"Rank {rank} Time taken: {time.time()-start:.2f} seconds.")
    
    if rank == 0:
        export_to_video(frames, "{}_async.mp4".format(file_name), fps=7)
        export_to_gif(frames, "{}_async.gif".format(rank))



if __name__ == "__main__":
    config = {
    "model_name": "stabilityai/stable-video-diffusion-img2vid-xt",
    "dtype": torch.float16,
    "strategy":"n2s1",
    "devices": ["cuda:0", "cuda:1"],
    "seed": 20,
    "step": 50,
    "time_shift":False,
    "warm_up":2,
    }

    size = len(config["devices"])
    processes = []
    try:
        mp.set_start_method('spawn', force=True)
        print("spawned")
    except RuntimeError:
        pass

    for rank in range(size):
        p = mp.Process(target=run_inference, args=(rank, size, config))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()