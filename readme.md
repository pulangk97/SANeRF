## Overview
Code implementation of **SANeRF**.  
![overview](/overview/overview.png)  
Comparisons:  
<img src="/overview/drums.gif" width="400" height="200" alt="drums"> <img src="/overview/lego.gif" width="400" height="200" alt="lego">

This code is implemented based on TriMipRF.   
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
PyTorch (1.13.1 + CUDA 11.6)  
nvdiffrast  
tiny-cuda-nn  
```
pip install -r requirements.txt
```

## DATA 
### nerf_synthetic dataset
Please download the nerf_synthetic data from the NeRF official Google Drive.

## Reproduce SANeRF's results
```
bash ./scripts/reproduce_sanerf.sh
```

## Reproduce baseline's results
replace the ``` method=sanerf ``` to  ``` method=trimiprf ``` , then run:  
```
bash ./scripts/reproduce_sanerf.sh
```


