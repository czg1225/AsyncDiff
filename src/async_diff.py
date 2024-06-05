import torch.distributed as dist
from .tools import ResultPicker
from .pipe_config import splite_model

class ModulePlugin(object):
    def __init__(self,module,  model_i, stride=1, run_mode=None) -> None:
        self.model_i, self.stride, self.run_mode = model_i, stride, run_mode
        self.module = module
        self.module.plugin = self
        self.init_state()
        self.inject_forward()
        self.rank = dist.get_rank()

    def init_state(self,warmup_n=1):
        self.warmup_n = warmup_n
        self.result_structure, self.cached_result = None, None
        self.infer_step = 0
    
    def cache_sync(self, async_flag):
        if self.infer_step >= self.warmup_n:
            dist.broadcast(self.cached_result, self.model_i, async_op=async_flag)

    def inject_forward(self):
        assert not hasattr(self.module, 'old_forward'), "Module already has old_forward attribute."
        module = self.module
        module.old_forward = module.forward
        
        def new_forward(*args, **kwargs):
            run_locally = self.run_mode == (self.model_i, self.infer_step%self.stride)
            if self.infer_step<self.warmup_n or run_locally:
                result = module.old_forward(*args, **kwargs)
                if self.infer_step + 1 >= self.warmup_n:
                    self.cached_result, self.result_structure = ResultPicker.dump(result)
            else:
                result = ResultPicker.load(self.cached_result, self.result_structure)
            self.infer_step += 1
            return result
        
        module.forward = new_forward

class AsyncDiff(object):
    def __init__(self, pipeline, model_n=2, stride=1, warm_up=1, time_shift=False):
        dist.init_process_group("nccl")
        if not dist.get_rank(): assert model_n + stride - 1 == dist.get_world_size(), "[ERROR]: The strategy is not compatible with the number of devices. (model_n + stride - 1) should be equal to world_size."
        self.model_n = model_n
        self.stride = stride
        self.warm_up = warm_up
        self.time_shift = time_shift
        self.pipeline = pipeline.to(f"cuda:{dist.get_rank()}")
        self.pipe_id = pipeline.config._name_or_path
        self.reformed_modules = {}
        self.reform_pipeline()

    def reset_state(self,warm_up=None):
        self.warm_up = warm_up or self.warm_up
        for each in self.reformed_modules.values():
            each.plugin.init_state(warmup_n=warm_up)


    def reform_module(self, module, module_id, model_i):
        run_mode = (dist.get_rank(), 0) if dist.get_rank() < self.model_n else (self.model_n -1, dist.get_rank() - self.model_n + 1)
        ModulePlugin(module, model_i, self.stride, run_mode)
        self.reformed_modules[(model_i, module_id)] = module
    
    def reform_unet(self):
        unet = self.pipeline.unet
        assert not hasattr(unet, 'old_forward'), "Unet already has old_forward attribute."
        unet.old_forward = unet.forward

        def unet_forward(*args, **kwargs):
            infer_step = self.reformed_modules[(0, 0)].plugin.infer_step
            if infer_step and infer_step%self.stride == 0:
                for each in self.reformed_modules.values():
                    each.plugin.cache_sync(False)
            if self.time_shift:
                infer_step = self.reformed_modules[0].plugin.infer_step
                if infer_step>=self.warm_up:
                    args = list(args)
                    args[1] = self.pipeline.scheduler.timesteps[infer_step-1]
            sample = unet.old_forward(*args, **kwargs)[0]
            if infer_step and infer_step%self.stride == 0:
                dist.broadcast(sample, self.model_n-1)
            return sample,

        unet.forward = unet_forward


    def reform_pipeline(self):
        models = splite_model(self.pipeline, self.pipe_id, self.model_n)
        for model_i, sub_model in enumerate(models):
            for module_id, module in enumerate(sub_model):
                self.reform_module(module, module_id, model_i)
        self.reform_unet()