
# pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16)
#transformer_blocks
#pos_embed
#norm_out
#proj_out
#adaln_single
#caption_projection

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os


def run_inference(rank, world_size, config):
    print(f"Rank {rank} is running.")

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29510'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if config["strategy"] == "n3s2":
        from pixart.n3s2 import AsyncDiffusionWorker
    elif config["strategy"] == "n2s1":
        from pixart.n2s1 import AsyncDiffusionWorker
    elif config["strategy"] == "n3s1":
        from pixart.n3s1 import AsyncDiffusionWorker
    elif config["strategy"] == "n4s1":
        from pixart.n4s1 import AsyncDiffusionWorker
    elif config["strategy"]== "n2s2":
        from pixart.n2s2 import AsyncDiffusionWorker


    pipeline = AsyncDiffusionWorker(
        config,
    )
    torch.cuda.set_device(
        config["devices"][rank]
    )  # this is necessary to make sure the correct device is set

    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    
    # warm up
    pipeline.reset_state()
    image = pipeline(
        config["prompt"],
        num_inference_steps=config["step"],
    ).images[0]
    
    for i in range(1):
        pipeline.reset_state(warm_up=config["warm_up"])
        torch.manual_seed(config["seed"])
        torch.cuda.manual_seed_all(config["seed"])
        start = time.time()
        image = pipeline(
            config["prompt"],
            num_inference_steps=config["step"],
        ).images[0]
        print(f"Rank {rank} Time taken: {time.time()-start:.2f} seconds.")
    image.save(f"output_rank{rank}.jpg")


if __name__ == "__main__":
    config = {
    "model_name": "PixArt-alpha/PixArt-XL-2-1024-MS",
    "dtype": torch.float16,
    "strategy":"n2s1",
    "devices": ["cuda:4","cuda:5"],
    "seed": 20,
    "step": 20,
    "time_shift":False,
    "warm_up":1,
    "prompt":"(fractal crystal skin:1.1) with( ice crown:1.4) woman, white crystal skin, (fantasy:1.3), (Anna Dittmann:1.3)",
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