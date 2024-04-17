import random
import numpy as np
import torch
import os
from loguru import logger
import cv2

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def downsample(img, factor, patch_size=-1, mode=cv2.INTER_AREA):
  """Area downsample img (factor must evenly divide img height and width)."""
  sh = img.shape
  max_fn = lambda x: max(x, patch_size)
  out_shape = (max_fn(sh[1] // factor), max_fn(sh[0] // factor))
#   print(out_shape)
  img = cv2.resize(img, out_shape, mode)
  return img

