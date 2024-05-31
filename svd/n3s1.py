import torch
from diffusers import StableVideoDiffusionPipeline
import torch.distributed as dist
from result_picker import ResultPicker

class Communicator(object):
    def __init__(self, run_locally, module,mode) -> None:
        self.run_locally = run_locally
        self.module = module
        self.module.communicator = self
        self.result_structure, self.cached_result = None, None
        self.rank = dist.get_rank()
        self.mode = mode
        self.init_state()
        self.inject_forward()

    def init_state(self,already_warmed_up=1):
        self.already_warmed_up = already_warmed_up
        self.result_structure, self.cached_result = None, None
        self.infer_step = 0

    def cache_sync(self,i,async_flag=False):
        if self.already_warmed_up<=0:
            dist.broadcast(self.cached_result, i, async_op=async_flag)

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
                self.infer_step += 1
                return result
            return new_forward
        module.forward = get_new_forward()

class AsyncDiffusionWorker(object):
    def __init__(self, config):
        self.config = config
        self.reformed_modules = {}
        self.ranklist = []
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
        Communicator(dist.get_rank()==mode, module,mode)
        self.reformed_modules[module_id] = module
    
    def reform_pipeline(self):
        unet = self.pipeline.unet

        if self.config["model_name"] == "stabilityai/stable-video-diffusion-img2vid-xt":


            modules_2 = (
                unet.up_blocks[3],
                unet.conv_norm_out,
                unet.conv_out
            )
            modules_1 = (
                unet.down_blocks[3],
                unet.mid_block,
                unet.up_blocks[0],
                unet.up_blocks[1],
                unet.up_blocks[2],
            )
            modules_0 = (
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.down_blocks[2],
            )



        for i, module in enumerate(modules_0):
            self.reform_module(module, i, 0)
            self.ranklist.append(0)

        for i, module in enumerate(modules_1):
            self.reform_module(module, i+len(modules_0), 1)
            self.ranklist.append(1)
        
        for i, module in enumerate(modules_2):
            self.reform_module(module, i+len(modules_0)+len(modules_1), 2)
            self.ranklist.append(2)

        def get_new_forward(module):
            if not hasattr(module, 'old_forward'):
                module.old_forward = module.forward
            def unet_forward(*args, **kwargs):

                for each, i in zip(self.reformed_modules.values(),self.ranklist):
                    each.communicator.cache_sync(i)
                
                if self.time_shift:
                    infer_step = self.reformed_modules[0].communicator.infer_step
                    if infer_step>=self.warm_up:
                        args = list(args)
                        args[1] = self.pipeline.scheduler.timesteps[infer_step-1]

                sample = module.old_forward(*args, **kwargs)[0]
                dist.broadcast(sample, 2)
                return sample,
                
            return unet_forward
        
        self.pipeline.unet.forward = get_new_forward(self.pipeline.unet)