from PIL import Image
from deeplab import *

def run_model(fileName, model):
  """Inferences DeepLab model and return results"""
  try:
    original_im = Image.open(fileName)
  except IOError:
    print('Cannot retrieve image. Please check file path: ' + fileName)
    return

  print('running deeplab on image %s...' % fileName)
  # 추론
  resized_im, seg_map = model.run(original_im)
  return seg_map
