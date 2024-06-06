


<p align="center">
<img src="assets/logo-modified.png" width="20%"> <br>
</p>

<div align="center">
<h1>AsyncDiff</h1>
  <div align="center">
  <a href="https://opensource.org/licenses/Apache-2.0">
    <img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-4E94CE.svg">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-%3E=v1.7.1-EE4C2C.svg?style=flat-square" alt="PyTorch>=v1.7.1">
  </a>
  <a href="https://github.com/facebookresearch/llama">
    <img src="https://img.shields.io/badge/LLMs-LLaMA-FFB000.svg?style=flat-square" alt="LLaMA">
  </a>
  <a href="https://github.com/facebookresearch/llama">
    <img src="https://img.shields.io/badge/LLMs-Llama2-FAB093.svg?style=flat-square" alt="Llama-2">
  </a>
  <a href="https://github.com/lm-sys/FastChat">
    <img src="https://img.shields.io/badge/LLMs-Vicuna-924E7D.svg?style=flat-square" alt="Vicuna">
  </a>
  <a href="https://huggingface.co/docs/transformers/model_doc/bloom">
    <img src="https://img.shields.io/badge/LLMs-BLOOM-1A63BD.svg?style=flat-square" alt="BLOOM">
  </a>
  <a href="https://github.com/THUDM/ChatGLM-6B">
    <img src="https://img.shields.io/badge/LLMs-chatGLM-6082B6.svg?style=flat-square" alt="chatGLM">
  </a>
    <a href="https://github.com/baichuan-inc/Baichuan-7B">
    <img src="https://img.shields.io/badge/LLMs-Baichuan-78ac62.svg?style=flat-square" alt="Baichuan">
  </a>
</div>
<h3>Parallelizing Diffusion Models by Asynchronous Denoising<h3>
</div>








## 🔧 Quick Start

- Create environment：

  ```shell
  conda create -n asyncdiff python=3.9
  conda activate asyncdiff
  pip install -r requirements.txt
  ```
  Please ensure that the model that needs to be accelerated has been downloaded in your environment (SDXL, SD2.1, SD1.5, AnimateDiff, SVD)



### Inference：
Please adjust the following configuration parameters in running scripts (run_sd.py, run_sdxl.py, run_animate.py, run_svd.py).

```python
config = {
    "model_name": "stabilityai/stable-diffusion-2-1",
    "dtype": torch.float16,
    "strategy":"n3s2",
    "devices": ["cuda:0","cuda:1", "cuda:2", "cuda:3"],
    "seed": 20,
    "step": 50,
    "time_shift":False,
    "warm_up":9,
    }

```
#### Accelerate Stable Diffusion 2.1 or 1.5:
```python
python run_sd.py
```


#### Accelerate Stable Diffusion XL:
```python
python run_sdxl.py
```


#### Accelerate Animate Diffusion:
```python
python run_animate.py
```


#### Accelerate Stable Video Diffusion:
```python
python run_svd.py
```

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 scripts/sd_example.py
CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 --master-port 29511 scripts/sd_example.py
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 --master-port 29511 scripts/sd_example.py

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master-port 29511 runs.py

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master-port 29511 runs.py

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29511 runs.py

CUDA_VISIBLE_DEVICES=0,1,2 torchrun --nproc_per_node=3 --master-port 29511 runs.py

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master-port 29511 sd_example.py

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master-port 29511 sdxl_example.py --model_n 2 --stride 1 --warm_up 2

CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master-port 29511 svd_example.py --model_n 2 --stride 1 --warm_up 1

git remote set-url origin https://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@github.com/czg1225/AsyncDiff.git
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