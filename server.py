from flask import Flask, request, abort
import werkzeug
import inference
from deeplab import DeepLabModel
from trimLabel import trimLabel,checkVolumnOfLiquid
import os

# 특정 파일 추론 시
import time
import numpy as np
from PIL import Image


app = Flask(__name__)

@app.before_request
def limit_addr():
    if request.remote_addr != "210.218.145.162":
        abort(403)

# 기본 경로 POST 요청 처리
@app.route('/', methods=['POST'])
def handle_request():
    # request msg의 image형식 파일
    imagefile= request.files['image']
    # filename 안전하게 추출
    filename = werkzeug.utils.secure_filename(imagefile.filename) 
    print("Received image File name : "+ imagefile.filename)
    # image file로 save
    imagefile.save(filename) 

    # 추론 실행 (segmentation label 배열 반환)
    resized_im,seg_map = inference.run_model(filename, model) 

    # label 다듬는 후처리 진행
    # issuccess : 인식 후처리에서 성공/실패 , msg : 후처리 상황 메시지
    issuccess, seg, msg = trimLabel(filename,seg_map)
    ingName = request.form["ingName"]
    if not issuccess:
        return {"success":"false","msg":msg,"ingName":ingName}
    
    # check fluid
    requiredRatio = request.form["ratio"]
    print("요청 비율 : ",requiredRatio)
    # 요청 비율만큼 채워졌는 지 확인
    # lineLoc : 어플에 띄울 선 위치 , isCheckable : ratio 측정 성공여부 , ratioMsg : 해당 결과 설명문 , ratioStatus : 
    lineLoc, isCheckable, ratioMsg, ratioStatus = checkVolumnOfLiquid(seg,float(requiredRatio))

    # 추론에 사용한 사진 삭제
    if os.path.exists(filename):
        os.remove(filename)
    
    # ratio 성공여부에 따라 ratio 값을 다르게 해 어플 내 오류 방지
    if isCheckable:
        return {"ingName":ingName,"success":"true","segmap":lineLoc,"msg":msg, "ratio":"true", "ratioMsg":ratioMsg,"ratioStatus":ratioStatus}
    else :
        return {"ingName":ingName,"success":"true","segmap":lineLoc,"msg":msg, "ratio":"false", "ratioMsg":ratioMsg,"ratioStatus":ratioStatus}


if __name__ == "__main__":
    model=DeepLabModel('') # 서버 실행 시 모델 미리 로드

    app.run(host="0.0.0.0", port=8081, debug=True) # 서버 실행
    
    '''
    # 특정 파일 추론 시

    # Make LUT (Look Up Table) with your 3 colours
    LUT = np.zeros((4,3),dtype=np.uint8)
    LUT[0]=[0,0,0]
    LUT[1]=[0,255,0]
    LUT[2]=[0,0,255]
    LUT[3]=[255,0,0]  

    # 추론
    for i  in range(19,21) :
        filename="./savedModel/air/"+str(i)+".jpg"
        _,seg_map = inference.run_model(filename, model)
        issuccess, seg, msg = trimLabel(filename,seg_map)
        lineLoc, isCheckable, ratioMsg, ratioStatus = checkVolumnOfLiquid(seg,float(100))
        print(msg,ratioMsg)
        pixelmap=LUT[seg_map] # label to color mapping
        im = Image.fromarray(pixelmap)
        im.save("./savedModel/air/"+str(i)+"_22_trim.jpg")
    #file=open("./maptxt/glass_30196.txt","w")
    #np.savetxt(file, seg_map.astype(int), fmt='%i')
    #file.close()
    '''