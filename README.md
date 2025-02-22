<p align="center">
<img src="assets/logo-modified.png" width="23%"> <br>
</p>

<div align="center">
<h1>AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoising</h1>

  <div align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-4E94CE.svg">
  </a>
  <a href="https://arxiv.org/abs/2406.06911">
    <img src="https://img.shields.io/badge/Conference-NeurIPS'24-924E7D.svg" alt="Paper">
  </a>
  <a href="https://czg1225.github.io/asyncdiff_page/">
    <img src="https://img.shields.io/badge/Project-Page-FFB000.svg" alt="Project">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-%3E=v2.0.1-EE4C2C.svg" alt="PyTorch>=v2.0.1">
  </a>
</div>
</div>

> **AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoising**   
> [Zigeng Chen](https://github.com/czg1225), [Xinyin Ma](https://horseee.github.io/), [Gongfan Fang](https://fangggf.github.io/), [Zhenxiong Tan](https://github.com/Yuanshi9815), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)   
> [xML Lab](https://sites.google.com/view/xml-nus), National University of Singapore  
> 🥯[[Paper]](https://arxiv.org/abs/2406.06911)🎄[[Project Page]](https://czg1225.github.io/asyncdiff_page/) \
> Code Contributors: [Zigeng Chen](https://github.com/czg1225), [Zhenxiong Tan](https://github.com/Yuanshi9815)

<div align="center">
  <img src="assets/combined.png" width="100%" ></img>
  <br>
  <em>
      2.8x Faster on SDXL with 4 devices. Top: 50 step original (13.81s). Bottom: 50 step AsyncDiff (4.98s)
  </em>
</div>
<br>

<div align="center">
  <img src="assets/combined.gif" width="100%" ></img>
  <br>
  <em>
      1.8x Faster on AnimateDiff with 2 devices. Top: 50 step original (43.5s). Bottom: 50 step AsyncDiff (24.5s)
  </em>
</div>
<br>

### Updates
* :tada: **September 26, 2024**: Our AsyncDiff is accepted by NeurIPS 2024!
* 🚀 **August 14, 2024**: Now supporting Stable Diffusion XL Inpainting! The inference sample of accelerating SDXL Inpainting can be found at [run_sdxl_inpaint.py](https://github.com/czg1225/AsyncDiff/blob/main/examples/run_sdxl_inpaint.py).
* 🚀 **July 18, 2024**: Now supporting Stable Diffusion 3 Medium! The inference sample of accelerating SD 3 can be found at [run_sd3.py](https://github.com/czg1225/AsyncDiff/blob/main/examples/run_sd3.py).
* 🚀 **June 18, 2024**: Now supporting ControlNet! The inference sample of accelerating controlnet+SDXL can be found at [run_sdxl_controlnet.py](https://github.com/czg1225/AsyncDiff/blob/main/examples/run_sdxl_controlnet.py).
* 🚀 **June 17, 2024**: Now supporting Stable Diffusion x4 Upscaler! The inference sample can be found at [run_sd_upscaler.py](https://github.com/czg1225/AsyncDiff/blob/main/examples/run_sd_upscaler.py).
* 🚀 **June 12, 2024**: Code of AsyncDiff is released.

### Supported Diffusion Models:
- ✅ [Stable Diffusion 3 Medium](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
- ✅ [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- ✅ [Stable Diffusion 1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- ✅ [Stable Diffusion x4 Upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler) 
- ✅ [Stable Diffusion XL 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) 
- ✅ [Stable Diffusion XL Inpainting](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1) 
- ✅ [ControlNet](https://huggingface.co/docs/diffusers/using-diffusers/controlnet#text-to-image) 
- ✅ [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
- ✅ [AnimateDiff](https://huggingface.co/docs/diffusers/api/pipelines/animatediff)

## Introduction
We introduce **AsyncDiff**, a universal and plug-and-play diffusion acceleration scheme that enables model parallelism across multiple devices. Our approach divides the cumbersome noise prediction model into multiple components, assigning each to a different device. To break the dependency chain between these components, it transforms the conventional sequential denoising into an asynchronous process by exploiting the high similarity between hidden states in consecutive diffusion steps. Consequently, each component is facilitated to compute in parallel on separate devices. The proposed strategy significantly reduces inference latency while minimally impacting the generative quality.


![AsyncDiff Overview](assets/fig2.png)
Above is the overview of the asynchronous denoising process. The denoising model εθ is divided into four components for clarity. Following the warm-up stage, each component’s input is
prepared in advance, breaking the dependency chain and facilitating parallel processing.

## 🔧 Quick Start

### Installation
- Prerequisites

  NVIDIA GPU + CUDA >= 12.0 and corresponding CuDNN


- Create environment：

  ```shell
  conda create -n asyncdiff python=3.10
  conda activate asyncdiff
  pip install -r requirements.txt
  ```

### Usage Example
Simply add two lines of code to enable asynchronous parallel inference for the diffusion model.
```python
import torch
from diffusers import StableDiffusionPipeline
from asyncdiff.async_sd import AsyncDiff

pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", 
torch_dtype=torch.float16, use_safetensors=True, low_cpu_mem_usage=True)

async_diff = AsyncDiff(pipeline, model_n=2, stride=1, time_shift=False)

async_diff.reset_state(warm_up=1)
image = pipeline(<prompts>).images[0]
if dist.get_rank() == 0:
  image.save(f"output.jpg")
```
Here, we use the Stable Diffusion pipeline as an example. You can replace `pipeline` with any variant of the Stable Diffusion pipeline, such as SD 2.1, SD 1.5, SDXL, or SVD. We also provide the implementation of AsyncDiff for AnimateDiff in `asyncdiff.async_animate`.
* `model_n`: Number of components into which the denoising model is divided. Options: 2, 3, or 4.
* `stride`: Denoising stride of each parallel computing batch. Options: 1 or 2.
* `warm_up`: Number of steps for the warm-up stage. More warm-up steps can achieve pixel-level consistency with the original output while slightly reducing processing speed.
* `time_shift`: Enables time shifting. Setting `time_shift` to `True` can enhance the denoising capability of the diffusion model. However, it should generally remain `False`. Only enable `time_shift` when the accelerated model produces images or videos with significant noise.



## Inference
We offer detailed scripts in `examples/` for accelerating inference of SD 2.1, SD 1.5, SDXL, SD 3, ControNet, SD_Upscaler, AnimateDiff, and SVD using our AsyncDiff framework.

### 🚀 Accelerate Stable Diffusion XL:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --run-path examples/run_sdxl.py
```

### 🚀 Accelerate Stable Diffusion 2.1 or 1.5:
```python
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run --nproc_per_node=4 --run-path examples/run_sd.py
```

### 🚀 Accelerate Stable Diffusion 3 Medium:
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path examples/run_sd3.py
```

### 🚀 Accelerate Stable Diffusion x4 Upscaler:
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path examples/run_sd_upscaler.py
```

### 🚀 Accelerate SDXL Inpainting:
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path examples/run_sdxl_inpaint.py
```

### 🚀 Accelerate ControlNet+SDXL :
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path examples/run_sdxl_controlnet.py
```

### 🚀 Accelerate Animate Diffusion:
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path examples/run_animatediff.py
```

### 🚀 Accelerate Stable Video Diffusion:
```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 --run-path examples/run_svd.py
```

## Qualitative Results
Qualitative Results on SDXL and SD 2.1. More qualitative results can be found in our [paper](https://arxiv.org/abs/2406.06911).
![Qualitative Results](assets/qualitative.png)

![Qualitative Results](assets/qualitative2.png)

## Quantitative Results
Quantitative evaluations of **AsyncDiff** on three text-to-image diffusion models, showcasing various configurations. More quantitative results can be found in our [paper](https://arxiv.org/abs/2406.06911).
![Quantitative Results](assets/quantitative.png)

## Bibtex
```
@article{chen2024asyncdiff,
  title={AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoising},
  author={Chen, Zigeng and Ma, Xinyin and Fang, Gongfan and Tan, Zhenxiong and Wang, Xinchao},
  journal={arXiv preprint arXiv:2406.06911},
  year={2024}
}
```
