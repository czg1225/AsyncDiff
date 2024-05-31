import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import os


def run_inference(rank, world_size,config):
    print(f"Rank {rank} is running.")

    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29511'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    if config["strategy"] == "n3s2":
        from sd.n3s2 import AsyncDiffusionWorker
    elif config["strategy"] == "n2s1":
        from sd.n2s1 import AsyncDiffusionWorker
    elif config["strategy"] == "n3s1":
        from sd.n3s1 import AsyncDiffusionWorker
    elif config["strategy"] == "n4s1":
        from sd.n4s1 import AsyncDiffusionWorker
    elif config["strategy"]== "n2s2":
        from sd.n2s2 import AsyncDiffusionWorker


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
    
    for i in range(3):
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
    "model_name": "stabilityai/stable-diffusion-2-1",
    "dtype": torch.float16,
    "strategy":"n3s2",
    "devices": ["cuda:0","cuda:1","cuda:2","cuda:3"],
    "seed": 20,
    "step": 50,
    "time_shift":False,
    "warm_up":9,
    "prompt":'A kitten that is sitting down by a door',
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