import torch
from diffusers import StableDiffusionXLPipeline
import torch.distributed as dist
from result_picker import ResultPicker


class SubmoduleCommunicator(object):
    def __init__(self, mode, module) -> None:
        self.rank = dist.get_rank()
        self.mode = mode
        self.run_locally = (self.rank<2 and mode==1) or (self.rank==2 and mode==0)
        self.module = module
        self.module.communicator = self
        self.init_state()
        self.inject_forward()

    def init_state(self, already_warmed_up=3):
        self.already_warmed_up = already_warmed_up
        self.infer_step = 0
        self.result_structure = None
        self.remote_result = [None]

    def cache_sync(self,async_flag):
        if self.already_warmed_up<=0:
            if self.mode == 0:
                dist.broadcast(self.remote_result[0],src= 2,async_op=async_flag)
            elif self.mode == 1:
                dist.broadcast(self.remote_result[0],src= 1,async_op=async_flag)
        else:
            pass


    def inject_forward(self):
        module = self.module
        module.old_forward = module.forward
        
        def get_new_forward():
            def new_forward(*args, **kwargs):
                if self.already_warmed_up>0:
                    result = module.old_forward(*args, **kwargs)
                    result_tensor, self.result_structure = ResultPicker.dump(result)
                    if self.already_warmed_up<=1:
                        self.remote_result = [result_tensor.clone()]
                    self.already_warmed_up -= 1

                elif self.infer_step%2 == 1:

                    if self.rank == 0:
                        if self.run_locally:
                            result = module.old_forward(*args, **kwargs)
                            result_tensor, self.result_structure = ResultPicker.dump(result)
                        else:
                            result = ResultPicker.load(self.remote_result[0], self.result_structure)
                    elif self.rank == 1 or self.rank == 2:
                        if self.run_locally:
                            result = module.old_forward(*args, **kwargs)
                            result_tensor, self.result_structure = ResultPicker.dump(result)
                            self.remote_result[0] = result_tensor.clone()
                        else:
                            result = ResultPicker.load(self.remote_result[0], self.result_structure)
                else:
                    result = ResultPicker.load(self.remote_result[0], self.result_structure)

                self.infer_step += 1
                return result
                
            return new_forward
        module.forward = get_new_forward()

class AsyncDiffusionWorker(object):
    def __init__(self, config):
        self.rank = dist.get_rank()
        self.step = config["step"]
        self.config = config
        self.reformed_modules = {}
        self.load_pipeline(config["model_name"], config["dtype"])
        self.reform_pipeline()
        self.time_shift = config["time_shift"]

    def load_pipeline(self, pipeline_name, dtype=torch.float16):
        self.pipeline = StableDiffusionXLPipeline.from_pretrained(pipeline_name, torch_dtype=dtype, variant="fp16", use_safetensors=True, low_cpu_mem_usage=False)
        self.pipeline.safety_checker = lambda images, clip_input: (images, None)
        self.pipeline.to(self.config["devices"][self.rank])
    
    def reset_state(self,warm_up=2):
        for each in self.reformed_modules.values():
            each.communicator.init_state(already_warmed_up=warm_up)

    def __call__(self, *args, **kwargs):

        return self.pipeline(*args, **kwargs)

    def reform_module(self, module, module_id, mode):
        SubmoduleCommunicator(mode, module)
        self.reformed_modules[module_id] = module
    
    def reform_unet(self, unet):
        def get_new_forward(module):
            module.old_forward = module.forward
            def unet_forward(*args, **kwargs):

                infer_step = self.reformed_modules[0].communicator.infer_step
                if infer_step%2==0 and infer_step != 0:
                    for each in self.reformed_modules.values():
                        each.communicator.cache_sync(False)
                
                if (self.rank == 1 or self.rank == 2) and infer_step%2==1 and infer_step < self.step-1:
                    args = list(args)
                    args[1] = self.pipeline.scheduler.timesteps[infer_step+1]
                    sample = module.old_forward(*args, **kwargs)[0]
                else:
                    sample = module.old_forward(*args, **kwargs)[0]

                infer_step = self.reformed_modules[0].communicator.infer_step

                if infer_step%2==0:
                    dist.broadcast(sample, src=0, async_op=False)
                
                return (sample,)
            
            return unet_forward
        unet.forward = get_new_forward(unet)
    
    def reform_pipeline(self):
        unet = self.pipeline.unet
        
        if self.config["model_name"] == "stabilityai/stable-diffusion-xl-base-1.0":

            modules_0 = (
                unet.down_blocks[2],
        
                unet.mid_block,

                unet.up_blocks[0].resnets[0],
                unet.up_blocks[0].attentions[0],
                unet.up_blocks[0].resnets[1],
                unet.up_blocks[0].attentions[1],
            )

            modules_1 = (
                unet.conv_in,
                unet.down_blocks[0],
                unet.down_blocks[1],
                unet.up_blocks[1],
                unet.up_blocks[2],
                unet.conv_norm_out,
                unet.conv_out,

                unet.up_blocks[0].resnets[2],
                unet.up_blocks[0].attentions[2],
                *unet.up_blocks[0].upsamplers,
            )

 
        for i, module in enumerate(modules_0):
            self.reform_module(module, i, 0)

        for i, module in enumerate(modules_1):
            self.reform_module(module, len(modules_0) + i, 1)

        self.reform_unet(unet)
