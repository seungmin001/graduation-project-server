from flask import Flask, request, abort
import werkzeug
import inference
from deeplab import DeepLabModel
from trimLabel_210910 import trimLabel,checkAreaOfLiquid, checkVolumnOfLiquid
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

# 기본 경로 요청 처리
@app.route('/', methods=['POST'])
def handle_request():
    imagefile= request.files['image'] # request msg의 image형식 파일
    filename = werkzeug.utils.secure_filename(imagefile.filename) # filename 안전하게 추출
    print("Received image File name : "+ imagefile.filename)
    imagefile.save(filename) # image file로 save

    # 추론 실행 (segmentation class 배열 반환)
    resized_im,seg_map = inference.run_model(filename, model) 
    # return {"success":"true","segmap":seg_map.tolist(),"msg":"실험입니다.", "ratio":"false"}
    # 경계선 얻어내기
    # issuccess : 인식 후처리에서 성공/실패 , msg : status에 띄울 말
    issuccess, seg, msg = trimLabel(filename,seg_map)
    ingName = request.form["ingName"]
    if not issuccess:
        return {"success":"false","msg":msg,"ingName":ingName}
    
    # check fluid
    requiredRatio = request.form["ratio"]
    print("requiredRatio",requiredRatio)
    # isCheckable : ratio 측정 성공여부 및 넘김 여부 , ratioMsg : 해당 결과 설명문
    lineLoc, isCheckable, ratioMsg, ratioStatus = checkVolumnOfLiquid(seg,float(requiredRatio))
    
    if isCheckable:
        return {"ingName":ingName,"success":"true","segmap":lineLoc,"msg":msg, "ratio":"true", "ratioMsg":ratioMsg,"ratioStatus":ratioStatus}
    else :
        return {"ingName":ingName,"success":"true","segmap":lineLoc,"msg":msg, "ratio":"false", "ratioMsg":ratioMsg,"ratioStatus":ratioStatus}
    # segmentation 완료 후 저장한 사진 삭제
    if os.path.exists(filename):
        os.remove(filename)

    return {"success":"true","segmap":seg.tolist(),"msg":msg, "ratio":"false"} # json data형태로 response msg 보냄.

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