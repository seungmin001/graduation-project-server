import flask
import werkzeug
import inference
from deeplab import DeepLabModel

'''
# 특정 파일 추론 시
import time
import numpy as np
'''

app = flask.Flask(__name__)

# 기본 경로 요청 처리
@app.route('/', methods=['GET','POST'])
def handle_request():
    imagefile= flask.request.files['image'] # request msg의 image형식 파일
    filename = werkzeug.utils.secure_filename(imagefile.filename) # filename 안전하게 추출
    print("Received image File name : "+ imagefile.filename)
    imagefile.save(filename) # image file로 save

    seg_map = inference.run_model(filename, model) # 추론 실행 (segmentation class map 반환)

    return {"segmap":seg_map.tolist()} # json data형태로 response msg 보냄.

if __name__ == "__main__":
    model=DeepLabModel('') # 서버 실행 시 모델 미리 로드

    app.run(host="0.0.0.0", port=5000, debug=True) # 서버 실행
    
    '''
    # 특정 파일 추론 시
    time.sleep(5)

    seg_map = inference.run_model("./savedModel/image/glass_10158.jpg", model)
    time.sleep(5)
    file=open("./maptxt/glass_10158.txt","w")
    np.savetxt(file, seg_map.astype(int), fmt='%i')
    file.close()

    '''