## Overview
Code implementation of [**Spatial Annealing for Efficient Few-shot Neural Rendering (AAAI 2025)**](https://arxiv.org/abs/2406.07828) 
![overview](/overview/overview.png)  
Comparisons:  
<img src="/overview/drums.gif" width="400" height="200" alt="drums"> <img src="/overview/lego.gif" width="400" height="200" alt="lego">

This code is implemented based on [TriMipRF](https://github.com/wbhu/Tri-MipRF).   
## Installation
First, create a new sanerf environment:
```
conda create -n sanerf python==3.8
```
Next, activate the environment:
```
conda activate sanerf
```
Install the following dependency:  
[PyTorch](https://pytorch.org/get-started/locally/) (1.13.1 + CUDA 11.6)  
[nvdiffrast](https://nvlabs.github.io/nvdiffrast/)  
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)  
```
pip install -r requirements.txt
```

## DATA 
### nerf_synthetic dataset
Please download the nerf_synthetic dataset from the [NeRF official project](https://github.com/bmild/nerf).

## Reproduce SANeRF's results
```
bash ./scripts/reproduce_sanerf.sh
```  
The reproduced results have an error margin of 0.05dB.  
## Reproduce baseline's results
Replace ``` method=sanerf ``` with  ``` method=trimiprf ``` in "./scripts/reproduce_sanerf.sh" , then run:  
```
bash ./scripts/reproduce_sanerf.sh
```

## Citation
If you find our work useful, please cite it as  
```
@inproceedings{xiao2025spatial,
  title={Spatial Annealing for Efficient Few-shot Neural Rendering},
  author={Xiao, Yuru and Zhai, Deming and Zhao, Wenbo and Jiang, Kui and Jiang, Junjun and Liu, Xianming},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={8},
  pages={8691--8699},
  year={2025}
}
```


