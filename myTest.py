import inference
from deeplab import DeepLabModel
from trimLabel import trimLabel,checkVolumnOfLiquid
# 특정 파일 추론 시
import numpy as np
import cv2
import os

if __name__ == "__main__":
    model=DeepLabModel('') # 모델 미리 로드

    targetDir="./1202/"
    # find every jpg file in directory
    count=1
    for file in os.listdir(targetDir):
        if file.endswith('.jpg'):
            _,seg_map = inference.run_model(targetDir+file, model)
            # 추론 결과 txt 저장
            textName=targetDir+'txt/'+str(count)
            count+=1
            txtFile=open(textName+".txt","w")
            np.savetxt(txtFile, seg_map.astype(int), fmt='%i')
            txtFile.close()

            # 추론 결과 출력
            # img 불러오기
            img2=cv2.imread(targetDir+file)
            img2 = cv2.resize(img2, dsize=(513, 513), interpolation=cv2.INTER_AREA)
            # one-hot encoding 방식 사용해서 3차원으로 변환
            one_hot_targets2 = np.eye(3)[seg_map]
            one_hot_targets2 = np.array(one_hot_targets2, dtype=np.uint8)
            # 둘 조합하여 출력
            blended2 = cv2.addWeighted(img2, 0.8, one_hot_targets2*120, 0.4, 0, img2, 0)
            cv2.imshow('blended original', blended2)

            # 후처리 결과 출력
            issuccess, seg, msg = trimLabel(targetDir+file,seg_map)
            lineLoc, isCheckable, ratioMsg, ratioStatus = checkVolumnOfLiquid(seg,100.0)
            # img 불러오기
            img=cv2.imread(targetDir+file)
            img = cv2.resize(img, dsize=(513, 513), interpolation=cv2.INTER_AREA)
            # one-hot encoding 방식 사용해서 3차원으로 변환
            one_hot_targets = np.eye(3)[seg]
            one_hot_targets = np.array(one_hot_targets, dtype=np.uint8)
            # 둘 조합하여 출력
            blended = cv2.addWeighted(img, 0.8, one_hot_targets*120, 0.4, 0, img, 0)
            cv2.imshow('blended', blended)

            cv2.waitKey(0)
            cv2.destroyAllWindows()