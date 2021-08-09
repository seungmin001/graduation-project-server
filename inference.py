from PIL import Image
from deeplab import *
from canny import sharpness_improve, auto_canny
from collections import Counter

def run_model(fileName, model):
  """Inferences DeepLab model and return results"""
  try:
    original_im = Image.open(fileName)
  except IOError:
    print('Cannot retrieve image. Please check file path: ' + fileName)
    return

  # 추론
  print('running deeplab on image %s...' % fileName)
  resized_im, seg_map = model.run(original_im)

  return resized_im, seg_map

def lineDetect(img,seg):

  # canny 적용
  cArr=auto_canny(sharpness_improve(img))
  Image.fromarray(cArr).save("cArr.jpg")
  
  for row in cArr.tolist():
    counter=Counter(row)
    # print(counter.most_common())

  one=[] # seg:1 cup
  two=[] # seg:2 fluid
  # seg에서 1행 당 인식된 부분 저장
  for row in range(len(seg)):
    counter=Counter(seg[row])
    if counter[1]>0:
      one.append((counter[1],row)) # 한 줄 당 cup 인식 개수, row 위치
    if counter[2]>0:
      two.append((counter[1],row)) # 한 줄 당 fluid 인식 개수, row 위치


  # 컵 인식 부분 중 컵의 가장 긴 부분의 10%이상인 row를 컵의 시작점으로 가정
  if len(one)>0:
    cup_max= max(x for (x,y) in one)
    for x,y in one:
      # 10%이상 중 제일 처음 것 기준 seg에 그 줄을 모두 3으로 저장
      if x >= int(cup_max*0.1):
        for i in range(len(seg[y])):
          seg[y][i] = 3
        break
  
    # 물이 인식된 부분 중 컵 최대 길이의 10%이상인 첫 부분을 fluid 시작 부분으로 가정
    if len(two)>0:
      for x,y in two:
        # 10%이상 중 제일 처음 것 기준 seg에 그 줄을 모두 3으로 저장
        if x >= int(cup_max*0.1):
          for i in range(len(seg[y])):
            seg[y][i] = 3
          break

  return seg # TODO : 각 경계선 값 255(seg : 3)인 seg반환
