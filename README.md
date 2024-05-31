# AsyncDiff: Parallelizing Diffusion Models by Asynchronous Denoisin

<p align="center">
<img src="assets/logo.png" width="30%"> <br>
</p>

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
