import numpy as np
import cv2

upHalfHeight = 0
def adjustCup(label):
    global upHalfHeight
    l = label.copy()

    cup = np.array(np.where((l == 1) | (l == 2)))
    if cup[0].size==0:
        return l,False,"컵이 인식되지 않았습니다"
    fluid = np.array(np.where(l == 2))
    if fluid[0].size==0:
        return l, False, "액체가 인식되지 않았습니다"

    # 컵 상단 지름 계산

    cupTop = cup[0].min()
    cupTopOld =cupTop
    cupBottom = cup[0].max()
    # TODO: cupLeft, cupRight의 1/10? 안되면 삭제 추가
    upDia, upDiaIdx = 0, 0  # 윗면 가로 지름과 그 row값
    count, delete = 0, False
    for i in range(cupTop, int((cupBottom+cupTop*3)/4)+1):  # 컵 상단에서부터 컵의 25% 검사
        # 한 줄 씩 현재 행의 가로 길이구하기
        colInfo = np.array(np.where(l[i] == 1))  # 이 row에서 label 1인 col arr
        if(colInfo.size==0): # 영역 2개 이상일 경우 고려
            continue
        
        rowLen = colInfo.max()-colInfo.min()

        if(rowLen < 50):  # 따라지는 액체로 인해 컵이 위로 뾰족하게 인식되는 경우
            print("컵의 상단이 뾰족함 rowLen : ",rowLen)
            l[i][colInfo.min():colInfo.max()] = 0
            delete = True
        elif (rowLen - upDia > 2):  # 현재 row가 더 길 경우 지름 갱신
            upDia = rowLen
            upDiaIdx = i
            count = 0
            if(delete):
                cupTop = upDiaIdx
                delete = False
        else:  # 5번 작아지면 지름 확정
            count += 1
            if(count >= 5):
                break

    upHalfHeight = upDiaIdx - cupTop  # 윗면 세로 반지름
    if upHalfHeight/upDia > 0.25:  # 컵 윗면이 많이 나온 경우(컵 윗면의 세로반지름/가로반지름이 긴 경우)
        print("컵의 정면을 촬영해주세요")
    print("세로반지름 : ",upHalfHeight)

    # 컵 하단 지름 계산

    dnDia, dnDiaIdx = 0,0 # 컵 하단 가로 지름과 그 row 값
    count = 0 # count 초기화
    for i in range(cupBottom, int((cupBottom*2+cupTop)/3)-1, -1):# 컵 아래 33% 검사
        # 한 줄 씩 현재 행의 가로 길이구하기
        colInfo = np.array(np.where(l[i] == 1))  # 이 row에서 label 1인 col arr
        if(colInfo.size==0): # 영역 2개 이상일 경우 고려
            continue
        rowLen = colInfo.max()-colInfo.min()
        
        if(rowLen < 20 or rowLen - dnDia > 5):
            dnDia = rowLen
            dnDiaIdx = i
            count = 0

            if(dnDiaIdx <= int((cupTop+cupBottom*7)/8)): # 하단 가로 지름 위치가 너무 높은 경우 (컵이 둥근 경우)
                dnDiaIdx = int((cupTop+cupBottom*9)/10)
                print("하단 위치가 너무 높음" )
                break
        else:
            count+=1
            if(count>=5):
                break


    # 상단 컵 자르기
    for i in range(cupTopOld,upDiaIdx): # 윗면 위치 포함x
        colInfo = np.array(np.where((l[i] == 1) | (l[i] == 2)))  # 이 row에서 label 1,2인 col arr
        for j in colInfo:
            l[i][j]=0 # 배경으로 치환
    # 하단 컵 자르기
    # TODO : 액체 최하단 값과 컵 최하단의 일정 비율? (투명 컵 중 맨 밑이 유리벽인 경우때문)
    
    semiBottom = int((fluid[0].max()+dnDiaIdx )/2 ) if fluid[0].size>0 else dnDiaIdx
    for i in range(semiBottom+1, cupBottom+1): # 밑바닥위치 포함 x # dnDiaIdx
        colInfo = np.array(np.where((l[i] == 1) | (l[i] == 2)))  # 이 row에서 label 1,2인 col arr
        for j in colInfo:
            l[i][j]=0 # 배경으로 치환

    return l,True,"good"


def adjustFluid(label):
    global upHalfHeight
    labelCopy = label.copy()
    fluid = np.array(np.where(labelCopy == 2))
    # 액체 없으면 인자 그대로 반환하며 종료
    if fluid[0].size == 0:
        return label, False, "액체가 인식되지 않았습니다"
    fluidTop = fluid[0].min()  # 행 값 중 최소값

    cup = np.array(np.where((labelCopy == 1) | (labelCopy == 2)))
    fluidBottom = cup[0].max() # 컵의 최하단을 액체의 밑바닥으로 변경

    # 가장 긴 행 기준 조금만 낮추고 진행? (위 동그란 부분 없애는 뜻으로)
    # np.bincount(arr) : 주어진 배열 값 중 나타난 횟수를 값과 같은 index에 저장
    binC=np.bincount(fluid[0])
    binC=np.argpartition(binC, -20)[-20:] # 상위 20개 값 추출
    longestRow=int(np.min(binC)) # 가장 상단 행 선택

    # TODO: longestRow and fluidTop의 일정비율로 낮추기? 
    semiTop = int((fluidTop+longestRow*2)/3)
    for i in range(fluidTop,semiTop): # 가장 긴 행 만큼 낮추기 # longestRow
        colInfo = np.array(np.where(labelCopy[i] == 2))  # 이 row에서 label 2인 col arr
        if colInfo.size==0: # 
            continue
        for j in colInfo: # 컵 치환
            labelCopy[i][j] = 1
    for i in range(semiTop,fluidBottom+1): # 컵 최하단까지 액체 늘리기
        colInfo = np.array(np.where(labelCopy[i] == 1))  # 이 row에서 label 1인 col arr
        if colInfo.size==0:
            continue
        for j in range(colInfo.min(),colInfo.max()+1): # 액체로 변환, min max로 해야 구멍이 안 생김
            labelCopy[i][j] = 2
    
    return labelCopy, True, "액체 인식 성공"


def readLabel(img_name):
    # black_background / test # video1_3 / white_background
    with open('./PostProcessingCode/txt/sample/test/'+img_name+'.txt', 'r') as file:
        output = [line.strip().split(' ') for line in file.readlines()]

    # numpy array로 변환
    output = np.array(output)
    # 문자 -> int로 타입 변환
    output = output.astype('uint8')
    return output


def showBlendedImage(label, img_name, num):
    # 0 -> [1,0,0]
    # 1 -> [0,1,0]
    # 2 -> [0,0,1]

    # one-hot encoding 방식 사용해서 3차원으로 변환
    one_hot_targets = np.eye(3)[label]
    one_hot_targets = np.array(one_hot_targets, dtype=np.uint8)
    # img 읽기
    img = cv2.imread(
        './PostProcessingCode/image/sample/test/'+img_name+'.jpg')
    img = cv2.resize(img, dsize=(513, 513), interpolation=cv2.INTER_AREA)
    # 둘 조합하여 출력
    blended = cv2.addWeighted(img, 0.7, one_hot_targets*120, 0.3, 0, img, 0)
    cv2.imshow('blended'+num, blended)
    # cv2.imwrite('video1_3.jpg',one_hot_targets*120) # 확인용


def experimentTxt():
    while(True):
        s = input("filename : ")
        if s == "q":
            return
        label = readLabel(s)
        newLabel = adjustCup(label)
        
        newLabel = adjustFluid(newLabel)
        

        showBlendedImage(label, s, "1")
        showBlendedImage(newLabel,s,"2")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":

    experimentTxt()
