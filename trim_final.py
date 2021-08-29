import cv2
import numpy as np
from math import hypot

def get_dist(point_a, point_b):
    x1, y1 = point_a
    x2, y2 = point_b
    return hypot(x1-x2, y1-y2)

def readLabel(img_name):
    # 파일 열어 numpy array로 변환하는 함수
    # with open('./txt/'+img_name+'.txt', 'r') as file:
    #     output = [line.strip().split(' ') for line in file.readlines()]
    with open('./txt/210826/'+img_name+'_clahe.txt', 'r') as file:
        output = [line.strip().split(' ') for line in file.readlines()]

    # numpy array로 변환
    output = np.array(output)

    # 문자 -> int로 타입 변환
    output = output.astype('uint8')
    return output

def arr2img(arr):
    # 0 -> [1,0,0]
    # 1 -> [0,1,0]
    # 2 -> [0,0,1]

    # one-hot encoding 방식 사용해서 3차원으로 변환
    one_hot_targets = np.eye(3)[arr]

    return np.array(one_hot_targets, dtype=np.uint8)


def auto_canny(image, sigma=0.33):
    # Compute the median of the single channel pixel intensities
    v = np.median(image)
    # Apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(image, lower, upper)


def blend2img(img1, img2, mode):
    blended = cv2.addWeighted(img1, 0.8, img2, 0.4, 0, img1, mode)
    # cv2.imshow('blended', blended)
    return blended


def findMostPoint(cnt):
    cnt = np.asarray(cnt)
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return leftmost, rightmost, topmost, bottommost


def findMiddlePoint(cnt, middleX):
    cnt_x = cnt[:, :, 0]
    cnt_middle = np.where(cnt_x == [middleX])

    # index[top bottom]
    tmp = [0, 0]
    if(cnt[cnt_middle[0][0]][0][1] > cnt[cnt_middle[0][1]][0][1]):
        tmp[0] = cnt_middle[0][1]
        tmp[1] = cnt_middle[0][0]
    else:
        tmp[0] = cnt_middle[0][0]
        tmp[1] = cnt_middle[0][1]

    # find middle top&bottom point form contour (for this, have to use cv2.CHAIN_APPROX_NONE)
    return tuple(cnt[tmp[0]][0]), tuple(cnt[tmp[1]][0])


def trimFluid(line, point, label, cup_upper_height):
    # approx_line의 중간 10점 정도의 평균점 찾아 label 수정
    # + 컵과 액체 상단 cup_upper_height만큼 지우기
    print('trim fluid')
    label_copy = label.copy()

    # 선 상의 점과 point 사이의 거리가 가까운 순으로 5개(거리!!)
    distances = abs(line[:, 0, 0]-[point[0]])+abs(line[:, 0, 1]-[point[1]])
    near_distances = np.unique(distances)[:5]

    # point와 가까운 거리(near_distances)를 갖는 점들을 near_points에 append
    near_points = np.zeros(shape=(0,), dtype=int)
    for distance in near_distances:
        tmp = np.where(distances == distance)[0]
        near_points = np.append(near_points, np.array(tmp))

    # [0]: 행 [1]: 열 ?
    fluid = np.array(np.where(label == 2))
    left = fluid[1].min()
    right = fluid[1].max()
    top = fluid[0].min()
    bottom = fluid[0].max()

    # point와 가까운 점들의 높이로 평균(correction_height) 계산
    sum, count = 0, 0
    for i in near_points:
        sum = sum + line[int(i), 0, 1]
        count = count+1
    correction_height = int(sum/count)

    # correction_height까지 액체 label 변경. 이 때 컵 윗면의 세로 반지름(cup_upper_height)만큼은 제외
    label[correction_height + cup_upper_height:top +
          int((bottom-top)*0.4), left:right+1] = 2
    # correction_height이 기존 액체 label 상단 점보다 낮게 나왔을 경우(값이 클 경우) 위를 컵 label로 지움
    label[np.array(np.where(label == 1))[0].min()          :correction_height + cup_upper_height, left:right+1] = 1
    # 액체와 마찬가지로 컵 label도 컵 윗면 세로 반지름(cup_upper_height)만큼 제외
    label[:np.array(np.where(label == 1))[0].min() +
          cup_upper_height, :] = 0

    # cup, fluid 값 변경시킨 것이 배경 침범할 수 있으므로(좌우를 액체 가장 왼쪽,오른쪽을 기준으로 해서) 배경값들만 기존 값으로 변경
    label[np.where(label_copy == 0)] = 0

    return label


def modifyMask(mask, size):
    abs_mask = abs(size)
    left_tmp = np.roll(mask, -1*abs_mask, axis=1)
    right_tmp = np.roll(mask, abs_mask, axis=1)
    top_tmp = np.roll(mask, -1*abs_mask, axis=0)
    bottom_tmp = np.roll(mask, abs_mask, axis=0)

    if(size < 0):
        mask = mask & left_tmp
        mask = mask & right_tmp
        mask = mask & top_tmp
        mask = mask & bottom_tmp
    else:
        mask = mask | left_tmp
        mask = mask | right_tmp
        mask = mask | top_tmp
        mask = mask | bottom_tmp

    return mask


def checkCupNum(contours_cup, label_012):
    # 컵 segmentation 개수 확인 후 2개 이상일 경우 label_012에서 지움
    if(len(contours_cup) < 4):
        # 컵이 여러 개 인식되지 않은 경우
        return False, label_012
    else:
        # 컵이 여러 개 인식된 경우
        M = np.empty((0), dtype=int)
        for i in contours_cup:
            M1 = cv2.moments(i)
            if(M1['m00'] == 0):
                M = np.append(M, np.array([500]), axis=0)
                continue
            x = int(M1['m10']/M1['m00'])
            y = int(M1['m01']/M1['m00'])
            M = np.append(M, np.array([int(get_dist([x, y], [250, 250]))]), axis=0)

        # 거리순으로 인덱스 정렬 후 중심과 가장 가까운 contour의 좌우상하값 구하기
        idx = M.argsort()
        cnt_cup = np.array(contours_cup[idx[0]]).reshape(
            len(contours_cup[idx[0]]), 2)
        left = cnt_cup[:, 0].min()
        right = cnt_cup[:, 0].max()
        top = cnt_cup[:, 1].min()
        bottom = cnt_cup[:, 1].max()

        # 컵 주위 5픽셀 제외 나머지 label값 0으로 변경
        label_012[:top-5, :] = 0       # 상
        label_012[bottom+5:, :] = 0    # 하
        label_012[:, :left-5] = 0       # 좌
        label_012[:, right+5:] = 0       # 우

        return True, label_012


def trimFluidFollowCup(label, cnt_cup):
    # 액체 label을 컵 따라 매끈하게 만드는 함수
    # + 액체 밑면 중심을 통과하도록 평평하게 만들며 컵의 밑면도 지움

    # 컵의 밑에서 위로 올라가며 양 끝점의 거리 차를 비교했을 때 해당 값이 작아질 경우가 밑면의 지름이라 간주
    # lower_diameter: 밑면의 지름, lower_diameter_idx: 밑면 지름의 높이값 idx
    # 이 때 컵이 둥근 경우 지름이 계속 변하므로 일정 높이를 지나도 값이 바뀔 경우 아래를 지우지 않도록 함(lower_diameter_idx = 512)
    lower_diameter, lower_diameter_idx = 0, 0
    cnt_cup_copy = cnt_cup.reshape(len(cnt_cup), 2)
    _, _, cup_top, cup_bottom = findMostPoint(cnt_cup)
    for idx in range(cup_bottom[1], int(cup_top[1]/6*2+cup_bottom[1]/6*4)+1, -1):
        cup_row = cnt_cup_copy[np.where(cnt_cup_copy[:, 1] == idx)]
        diameter = cup_row[:, 0].max() - cup_row[:, 0].min()
        if(diameter-lower_diameter > 2):
            lower_diameter = diameter
            lower_diameter_idx = idx
            if(lower_diameter_idx <= int(cup_top[1]/10+cup_bottom[1]/10*9)):
                lower_diameter_idx = 512
                break
        else:
            break

    # 컵 label이 포함된 행(cup_width)에서 액체가 있는 행을 for문으로 돌아가며 값 변경
    fluid_height = np.where(label == 2)[0]
    fluid_height = np.unique(fluid_height)
    for i in fluid_height:
        cup_width = np.array(np.where(label[i] == 1))
        if len(cup_width[0]) == 0:
            continue
        elif i >= lower_diameter_idx:
            # 유효한 컵 밑면의 지름(lower_diameter_idx)을 구한 경우 그보다 밑의(값이 큰) 액체는 컵으로 변환
            label[i, cup_width.min():cup_width.max()+1] = 1
        else:
            # 컵 액체로 꽉 찬 경우(len(cup_width[0]) < 1) 제외
            label[i, cup_width.min():cup_width.max()+1] = 2

    # if want to remove the bottom of the cup
    cup_height = np.where(label == 1)[0]
    cup_height = np.unique(cup_height)
    if(cup_height.max() > fluid_height.max()):
        for i in range(fluid_height.max()+1, cup_height.max()+1):
            label[i, :] = 0

    return label


def trimLabel(image_name,seg_map):
    # 이미지 읽기
    img = cv2.imread(image_name)
    img = cv2.resize(img, dsize=(513, 513),
                     interpolation=cv2.INTER_AREA)
    img_copy = img.copy()

    # 라벨 읽기
    label_012 = seg_map # readLabel(image_name)
    label = arr2img(label_012)
    label_gray = cv2.cvtColor(label*120, cv2.COLOR_RGB2GRAY)

    # 액체 있는 경우(!0) 액체 trim
    if(np.count_nonzero(label_gray == 14) != 0):

        # 선명도 올리기
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))

        # canny
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        small_dst = cv2.resize(gray, dsize=(128, 128),
                               interpolation=cv2.INTER_AREA)
        small_dst = clahe.apply(small_dst)
        small_canny = auto_canny(small_dst)
        small_canny_enlarge = cv2.resize(small_canny, dsize=(513, 513),
                                         interpolation=cv2.INTER_AREA)

        # cup만 segmentation하도록 label값 조정
        # cup: 70, liquid: 14, background: 36
        label_cup = np.where(label_gray == 14, 70, label_gray)

        # cup segmentation canny -> contour
        label_cup_canny = auto_canny(label_cup)
        contours_cup, _ = cv2.findContours(
            label_cup_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # label_cup에서 컵이 있는 부분만 True, 나머지는 false로 mask 생성
        label_cup_mask = label_cup == 70
        # mask의 컵 부분(True/False) 축소/확대시키기
        reduce_label_cup_mask = modifyMask(label_cup_mask, -10)

        # cup 개수 확인 후 2개 이상일 경우 중심에 가장 가까운 contour 주변 5픽셀 제외하고 0으로 지움
        many_cup, label_012 = checkCupNum(contours_cup, label_012)

        # label 변경 시(컵2개이상인경우) contour 다시 구함
        if(many_cup):
            return False, [], "컵을 하나만 비치해주세요"

        # 가장 긴 contour를 cup의 contour로 가정
        cnt_cup = []
        for i in contours_cup:
            if (len(i) >= len(cnt_cup)):
                cnt_cup = i

        # liquid만 segmentation하도록 label값 조정
        # cup mask의 윗면 일괄 false 변환에 fluid_middle_top 필요
        # 액체 segmentation canny -> contour
        label_fluid = np.where(label_gray == 70, 36, label_gray)
        label_fluid_canny = auto_canny(label_fluid)
        contours_fluid, _ = cv2.findContours(
            label_fluid_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 가장 긴 contour를 liquid의 contour로 가정
        cnt_fluid = []
        for i in contours_fluid:
            if (len(i) >= len(cnt_fluid)):
                cnt_fluid = i

        # 액체의 극점, middle_top/bottom 찾기 (이 때, middle_top[1] < middle_bottom[1])
        fluid_left, fluid_right, _, _ = findMostPoint(
            cnt_fluid)
        fluid_middle_top, _ = findMiddlePoint(
            cnt_fluid, int((fluid_left[0]+fluid_right[0])/2))

        # cup mask의 윗면 일괄 false 변환.
        # 컵에 액체가 많이 따라진 경우는 다루지 않을 확률이 높으며 윗면 edge가 많이 남으면 결과에 부정적 영향을 줌
        # 컵의 위에서 아래로 내려가며 양 끝점의 거리 차를 비교했을 때 해당 값이 작아질 경우가 윗면의 지름이라 간주
        # upper_diameter: 윗면의 지름, upper_diameter_idx: 윗면 지름의 높이값 idx
        cup_left, cup_right, cup_top, cup_bottom = findMostPoint(cnt_cup)
        upper_diameter, upper_diameter_idx = 0, 0
        cnt_cup_copy = cnt_cup.reshape(len(cnt_cup), 2)
        for idx in range(cup_top[1], int(cup_bottom[1]/4 + cup_top[1]/4*3)+1):
            cup_row = cnt_cup_copy[np.where(cnt_cup_copy[:, 1] == idx)]
            diameter = cup_row[:, 0].max() - cup_row[:, 0].min()
            if(diameter-upper_diameter > 2):
                upper_diameter = diameter
                upper_diameter_idx = idx
            else:
                break

        cup_upper_height = upper_diameter_idx - cup_top[1]

        # print('cup_upper_height: ', cup_upper_height)
        if(cup_upper_height > 20):
            # 컵 윗면이 많이 나온 경우
            if(int((cup_right[1]+cup_left[1])/2) > fluid_middle_top[1]):
                # 컵 윗면의 중심이 fluid_middle_top보다 아래 -> 컵의 1/10 지움
                limit = int(cup_top[1] + (cup_bottom[1]-cup_top[1])/10)
            elif(int((cup_right[1]+cup_left[1])/2) + cup_upper_height > fluid_middle_top[1]):
                # 컵 윗면의 밑점이 fluid_middle_top보다 아래 -> 컵 윗면의 반 지움
                limit = int((cup_right[1]+cup_left[1])/2)
            else:
                # 컵 윗면의 밑점이 fluid_middle_top보다 위 -> 컵 윗면 지움
                limit = int((cup_right[1]+cup_left[1])/2) + cup_upper_height
        else:
            # 컵 윗면이 많이 나오지 X (정면에 가까움) -> 컵 상단의 20% 지움
            limit = int(cup_top[1] + (cup_bottom[1]-cup_top[1])/5)

        reduce_label_cup_mask[:limit, :] = False

        # small_canny_enlarge에서 컵이 있는 부분만 남기고 나머지는 지움. 이 때 1차원 되므로 reshape해준다.
        filtered_canny = np.where(
            reduce_label_cup_mask, small_canny_enlarge, 0)
        filtered_canny = np.reshape(filtered_canny, (513, 513))

        # cv2.imshow('canny', filtered_canny)

        # 컵부분만 남긴 edge들을 contour로 변환. 각 edge의 길이 비교를 위함.
        contours_filtered_cup, _ = cv2.findContours(
            filtered_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # ///////////////////////////////////////////////////////////////////////////////////////////
        if(len(contours_filtered_cup) < 5):
            # 비교 가능한 액체의 edge가 5개도 되지 않을 경우 error 코드 return (return 값 바꿔야함!!!)
            return False, [], "추론에 실패하였습니다"
        # ///////////////////////////////////////////////////////////////////////////////////////////

        # canny contour에서 면적 구해 append(길이 가장 긴 contour 찾기 위함!)
        i = 0
        contours_filtered_cup_area = np.empty((0, 2), dtype=float)
        for cnt_ in contours_filtered_cup:
            area = cv2.contourArea(cnt_)
            # print(i, '면적:', area)
            contours_filtered_cup_area = np.append(
                contours_filtered_cup_area, np.array([[i, area]]), axis=0)
            i = i+1

        # 구한 면적만 가지고 sort
        contours_filtered_cup_area_sort = contours_filtered_cup_area[contours_filtered_cup_area[:, 1].argsort(
        )]

        # 길이가 가장 긴(면적이큰) edge 5개의 중심점을 구해 M에 저장
        M = np.empty((0, 3), dtype=int)
        for i in range(5):
            M1 = cv2.moments(contours_filtered_cup[int(
                contours_filtered_cup_area_sort[-1*(i+1)][0])])
            x = int(M1['m10']/M1['m00'])
            y = int(M1['m01']/M1['m00'])
            M = np.append(M, np.array([[-1*(i+1), x, y]]), axis=0)

        # 액체의 middle_top과 가장 가까운 중심점을 갖는 edge 찾기
        tmp = []
        for i in range(5):
            tmp = np.append(tmp, get_dist(M[i][1:], fluid_middle_top))

        approx_line = contours_filtered_cup[int(
            contours_filtered_cup_area_sort[M[tmp.argmin()][0]][0])]

        # approx_line의 중간 10점 정도의 평균점 찾아 label 수정
        # + 컵과 액체 상단 cup_upper_height만큼 지우기
        label_012 = trimFluid(approx_line, fluid_middle_top,
                              label_012, cup_upper_height)
        # 액체 label을 컵 따라 매끈하게 만드는 함수
        # + 액체 밑면 중심을 통과하도록 평평하게 만들며 컵의 밑면도 지움
        label_012 = trimFluidFollowCup(label_012, cnt_cup)

    # 각 라벨 별 픽셀 개수 출력
    label_count = np.unique(label_012, return_counts=True)
    print(label_count)

    blended = cv2.addWeighted(img_copy, 0.8, label*120, 0.4, 0, img, 0)
    # cv2.imshow('blended_original', blended)

    return True, label_012, "추론에 성공하였습니다."
