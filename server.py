import flask
import werkzeug
import inference
from deeplab import DeepLabModel
from trimLabel_210910 import trimLabel,checkAreaOfLiquid, checkVolumnOfLiquid
import os
'''
# 특정 파일 추론 시
import time
import numpy as np
from PIL import Image
'''

app = flask.Flask(__name__)

# 기본 경로 요청 처리
@app.route('/', methods=['POST'])
def handle_request():
    imagefile= flask.request.files['image'] # request msg의 image형식 파일
    filename = werkzeug.utils.secure_filename(imagefile.filename) # filename 안전하게 추출
    print("Received image File name : "+ imagefile.filename)
    imagefile.save(filename) # image file로 save

    # 추론 실행 (segmentation class 배열 반환)
    resized_im,seg_map = inference.run_model(filename, model) 
    # 경계선 얻어내기
    # seg=inference.lineDetect(resized_im,seg_map)
    issuccess, seg, msg = trimLabel(filename,seg_map)
    if not issuccess:
        return {"success":"false","msg":msg,"ratio":"false"}
    
    # check fluid
    # {"ratio":"true"/"false"}
    requiredRatio = flask.request.form["ratio"]
    print("requiredRatio",requiredRatio)
    if checkVolumnOfLiquid(seg,float(requiredRatio)):
        return {"success":"true","segmap":seg.tolist(),"msg":"비율 만족! 다음 단계로 진행하세요", "ratio":"true"}
    else :
        return {"success":"true","segmap":seg.tolist(),"msg":msg, "ratio":"false"}
    # segmentation 완료 후 저장한 사진 삭제
    if os.path.exists(filename):
        os.remove(filename)

    return {"success":"true","segmap":seg.tolist(),"msg":msg, "ratio":"false"} # json data형태로 response msg 보냄.

if __name__ == "__main__":
    model=DeepLabModel('') # 서버 실행 시 모델 미리 로드

    app.run(host="0.0.0.0", port=8081, debug=True) # 서버 실행
    
    '''
    # 특정 파일 추론 시
    time.sleep(5)

    img=Image.open("./savedModel/image/glass_30196.jpg")
    img2=img.convert('RGB').resize((513,513),Image.ANTIALIAS)
    seg_map = inference.run_model(img2, model)
    print(len(seg_map), len(seg_map[0]))
    time.sleep(5)
    file=open("./maptxt/glass_30196.txt","w")
    np.savetxt(file, seg_map.astype(int), fmt='%i')
    file.close()
    '''