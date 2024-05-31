import torch
from diffusers import StableVideoDiffusionPipeline
import torch.distributed as dist
from result_picker import ResultPicker

class Communicator(object):
    def __init__(self, run_locally, module) -> None:
        self.run_locally = run_locally
        self.module = module
        self.module.communicator = self
        self.init_state()
        self.inject_forward()
        self.rank = dist.get_rank()

    def init_state(self,already_warmed_up=1):
        self.already_warmed_up = already_warmed_up
        self.result_structure, self.cached_result = None, None
        self.infer_step = 0

    
    def cache_sync(self,async_flag):
        if self.already_warmed_up<=0:
            dist.broadcast(self.cached_result, self.rank if self.run_locally else 1-self.rank, async_op=async_flag)

    def inject_forward(self):

        module = self.module
        module.old_forward = module.forward
        
        def get_new_forward():
            def new_forward(*args, **kwargs):
                if self.already_warmed_up>0 or self.run_locally:
                    result = module.old_forward(*args, **kwargs)
                    if self.already_warmed_up <= 1:
                        self.cached_result, self.result_structure = ResultPicker.dump(result)
                    self.already_warmed_up -= 1
                else:
                    result = ResultPicker.load(self.cached_result, self.result_structure)
                # module.communicator.cache_sync(True)
                self.infer_step += 1
                return result
            return new_forward
        module.forward = get_new_forward()

class AsyncDiffusionWorker(object):
    def __init__(self, config):
        self.config = config
        self.reformed_modules = {}
        self.load_pipeline(config["model_name"], config["dtype"])
        self.reform_pipeline()
        self.time_shift = config["time_shift"]
        self.warm_up = 1

    def load_pipeline(self, pipeline_name, dtype=torch.float16):
        self.pipeline = StableVideoDiffusionPipeline.from_pretrained(pipeline_name, torch_dtype=dtype, variant="fp16", use_safetensors=True, low_cpu_mem_usage=False)
        self.pipeline.safety_checker = lambda images, clip_input: (images, None)
        self.pipeline.to(self.config["devices"][dist.get_rank()])

    def reset_state(self,warm_up=1):
        self.warm_up = warm_up
        for each in self.reformed_modules.values():
            each.communicator.init_state(already_warmed_up=warm_up)

    def __call__(self, *args, **kwargs):

        return self.pipeline(*args, **kwargs)

    def reform_module(self, module, module_id, mode):
        Communicator(dist.get_rank()==mode, module)
        self.reformed_modules[module_id] = module
    
    def reform_pipeline(self):
        unet = self.pipeline.unet
        
        if self.config["model_name"] == "stabilityai/stable-video-diffusion-img2vid-xt":

            modules_0 = (
                unet.conv_in,
                *unet.down_blocks,
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
            )

            modules_1 = (
                *unet.up_blocks[2:],
                unet.conv_norm_out,
                unet.conv_out
            )

 
        for i, module in enumerate(modules_0):
            self.reform_module(module, i, 0)

        for i, module in enumerate(modules_1):
            self.reform_module(module, i+len(modules_0), 1)

        def get_new_forward(module):
            if not hasattr(module, 'old_forward'):
                module.old_forward = module.forward
            def unet_forward(*args, **kwargs):
        
                for each in self.reformed_modules.values():
                    each.communicator.cache_sync(False)
                
                if self.time_shift:
                    infer_step = self.reformed_modules[0].communicator.infer_step
                    if infer_step>=self.warm_up:
                        args = list(args)
                        args[1] = self.pipeline.scheduler.timesteps[infer_step-1]

                sample = module.old_forward(*args, **kwargs)[0]
                dist.broadcast(sample, 1)
                return sample,
                

            return unet_forward
        
        self.pipeline.unet.forward = get_new_forward(self.pipeline.unet)