## Overview
Code implementation of **SANeRF**.  

![overview](/overview/overview.png)  

![drums](/overview/drums.gif)  

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
[PyTorch (1.13.1 + CUDA 11.6)](https://pytorch.org/get-started/locally/)  
[nvdiffrast](https://nvlabs.github.io/nvdiffrast/)  
[tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)
```
pip install -r requirements.txt
```

## DATA 

