from PIL import Image
from deeplab import *

def run_model(fileName, model):
  """Inferences DeepLab model and return results"""
  try:
    original_im = Image.open(fileName)
    # original_im=original_im.convert('RGB').resize((513,513),Image.ANTIALIAS)
  except IOError:
    print('Cannot retrieve image. Please check file path: ' + fileName)
    return

  # 추론
  print('running deeplab on image %s...' % fileName)
  resized_im, seg_map = model.run(original_im)
  original_im.close()
  return resized_im, seg_map

