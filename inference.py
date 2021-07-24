import numpy as np
from PIL import Image
import sys
# tensorflow_version 1.15 (gpu)
import tensorflow as tf
from deeplab import *



def run_visualization(url, model):
  """Inferences DeepLab model and visualizes result."""
  try:
    original_im = Image.open(url)
  except IOError:
    print('Cannot retrieve image. Please check url: ' + url)
    return

  print('running deeplab on image %s...' % url)
  # 추론
  resized_im, seg_map = model.run(original_im)
  return seg_map
  # vis_segmentation(resized_im, seg_map)
