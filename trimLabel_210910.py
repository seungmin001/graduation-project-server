import cv2
import numpy as np
from math import hypot

# fluid 의 근사선을 찾기 위해 중심점 위치를 비교할 긴 edge의 개수
FLUID_APPROX_EDGE_NUM = 3


def get_dist(point_a, point_b):
    x1, y1 = point_a
    x2, y2 = point_b
    return hypot(x1-x2, y1-y2)


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


def findMostPoint(cnt):  # cnt는 한 윤곽선 [0]점/[0]/[0,1]xy
    cnt = np.array(cnt)
    # 모든 x값에 대하여 x가 가장 작은 점 [x,y]
    leftmost = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmost = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmost = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommost = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return leftmost, rightmost, topmost, bottommost


def findMiddlePoint(cnt, middleX):
    cnt_x = cnt[:, :, 0]
    cnt_middle = np.where(cnt_x == [middleX])

    bottom = [0, 0]
    top = [512, 512]

    for i in cnt_middle[0]:
        # print(cnt[i][0][1])
        # print(bottom[1])
        if(cnt[i][0][1] <= top[1]):
            top = cnt[i][0]
        if(cnt[i][0][1] >= bottom[1]):
            bottom = cnt[i][0]

    # find middle top&bottom point form contour (for this, have to use cv2.CHAIN_APPROX_NONE)
    return tuple(top), tuple(bottom)


def trimFluid(line, point, label, cup_upper_height):
    # approx_line의 중간 10점 정도의 평균점 찾아 label 수정
    # + 컵과 액체 상단 cup_upper_height만큼 지우기
    # print('trim fluid')
    label_copy = label.copy()

    # 선 상의 점과 point 사이의 거리가 가까운 순으로 5개(거리!!) 거리 = x좌표 간 거리 + y좌표 간 거리
    distances = abs(line[:, 0, 0]-[point[0]])+abs(line[:, 0, 1]-[point[1]])
    near_distances = np.unique(distances)[:10]

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

    # correction_height 보다 기존 액체 label이 높을 경우 지움
    if(top < correction_height):
        label[top:correction_height, left:right+1] = 1

    # correction_height까지 액체 label 변경. 이 때 컵 윗면의 세로 반지름(cup_upper_height)만큼은 제외 >>>>> 일단 포함하기로 ? resize 더 작게하면 액체 뒷쪽 edge 지울 수도 있을 것 같은데 여기 예외처리 잘 해야할 듯.
    # 액체가 이미지의 반 이상 찬 경우 액체 윗면이 직선에,그렇지 않을 경우 컵의 윗면에 가까울 것이라 가정.
    # (correction_height > top) or (correction_height-cup_upper_height>top) 인 경우 액체 label 채워지지 X.
    # correction_height 부터 액체 상단 40%까지만 액체로 label 바꾸던 것 bottom까지 전부 바꾸는 것으로 변경.
    # 액체 label이 두 개 추론된 경우 가운데 쪼개짐 발생하던 것 해결하기 위함.
    # correction_height이 기존 액체 label 상단 점보다 낮게 나왔을 경우(값이 클 경우) 위를 컵 label로 지움 >>>> 윗 내용따라 포함하므로 + ->= - 로 변경
    correction_height = correction_height + cup_upper_height
    if(correction_height < bottom):
        label[correction_height:bottom+1, left:right+1] = 2
        label[np.array(np.where(label == 1))[0].min()              :correction_height, left:right+1] = 1
    else:
        label[bottom:correction_height+1, left:right+1] = 2
        label[np.array(np.where(label == 1))[0].min():bottom, left:right+1] = 1

    # 컵 label 컵 윗면 세로 반지름(cup_upper_height)만큼 제외
    label[:np.array(np.where(label == 1))[0].min() +
          cup_upper_height, :] = 0

    # cup, fluid 값 변경시킨 것이 배경 침범할 수 있으므로(좌우를 액체 가장 왼쪽,오른쪽을 기준으로 해서) 배경값들만 기존 값으로 변경
    label[np.where(label_copy == 0)] = 0

    return label


def modifyMask(mask, size):
    # size: 양수일 경우 확대, 음수일 경우 축소시킴
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
            M = np.append(M, np.array(
                [int(get_dist([x, y], [250, 250]))]), axis=0)

        # 거리순으로 인덱스 정렬 후 중심과 가장 가까운 contour의 좌우상하값 구하기
        idx = M.argsort()
        cnt_cup = np.array(contours_cup[idx[0]]).reshape(
            len(contours_cup[idx[0]]), 2)
        left = cnt_cup[:, 0].min()
        right = cnt_cup[:, 0].max()
        top = cnt_cup[:, 1].min()
        bottom = cnt_cup[:, 1].max()

        # 컵 주위 5픽셀 제외 나머지 label값 0으로 변경
        if(top-5 >= 0):
            label_012[:top-5, :] = 0       # 상
        if(bottom+5 <= 512):
            label_012[bottom+5:, :] = 0    # 하
        if(left-5 >= 0):
            label_012[:, :left-5] = 0       # 좌
        if(right+5 <= 512):
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
    count = 0
    for idx in range(cup_bottom[1], int(cup_top[1]/3*1+cup_bottom[1]/3*2)+1, -1):   # 컵 아래 33%
        cup_row = cnt_cup_copy[np.where(cnt_cup_copy[:, 1] == idx)]
        diameter = cup_row[:, 0].max() - cup_row[:, 0].min()
        if(diameter < 20 or diameter-lower_diameter > 2):
            lower_diameter = diameter
            lower_diameter_idx = idx
            # 너무 높은 경우(컵이 둥근 경우)
            if(lower_diameter_idx <= int(cup_top[1]/8+cup_bottom[1]/8*7)):
                lower_diameter_idx = cup_top[1]/10+cup_bottom[1]/10*9
                break
        else:
            count += 1
            if(count >= 5):
                break

    # 컵 label이 포함된 행(cup_width)에서 액체가 있는 행을 for문으로 돌아가며 값 변경 >>>>> 이 부분을.. 수정해야할지? 액체 segmentation 추가 학습 후에 고려해볼 것.
    fluid_height = np.where(label == 2)[0]
    fluid_height = fluid_height.min()
    cup_height = np.where(label == 1)[0]
    cup_height = np.unique(cup_height)
    for i in cup_height:
        cup_width = np.array(np.where(label[i] == 1))
        if len(cup_width[0]) == 0:
            continue
        elif i >= fluid_height:
            if i >= lower_diameter_idx:
                # 유효한 컵 밑면의 지름(lower_diameter_idx)을 구한 경우 그보다 밑의(값이 큰) 액체는 컵으로 변환
                label[i, cup_width.min():cup_width.max()+1] = 1
            else:
                label[i, cup_width.min():cup_width.max()+1] = 2
        else:
            # 컵 액체로 꽉 찬 경우(len(cup_width[0]) < 1) 제외
            label[i, cup_width.min():cup_width.max()+1] = 1

    fluid_height = np.where(label == 2)[0]
    fluid_height = np.unique(fluid_height)
    # if want to remove the bottom of the cup >>>>> 요기도 추가 학습 후 고려해볼 것. 액체 추론이 제대로 안 될 경우 아래가 너무 많이 지워짐..
    cup_height = np.where(label == 1)[0]
    cup_height = np.unique(cup_height)

    if((len(fluid_height) > 0) and (len(cup_height) > 0)):
        if (cup_height.max() > fluid_height.max()):
            for i in range(fluid_height.max()+1, cup_height.max()+1):
                label[i, :] = 0
    return label


def checkAreaOfLiquid(label, ratio):
    # 컵과 액체의 면적을 파악하고 액체의 양이 원하는 비율만큼 채워졌는지 검사
    _, label_count = np.unique(label, return_counts=True)
    cup_area = label_count[1] + label_count[2]
    liquid_area = label_count[2]
    valid_cup_area = cup_area*0.8

    # print('Area ratio: ', liquid_area/valid_cup_area*100)

    if((ratio-2 <= liquid_area/valid_cup_area*100) and (ratio+2 >= liquid_area/valid_cup_area*100)):
        return True
    else:
        return False


def checkVolumnOfLiquid(label, ratio):
    # 컵과 액체의 면적을 파악하고 액체의 양이 원하는 비율만큼 채워졌는지 검사
    # np.where()[0]: row, np.where()[1]: col

    cup = np.array(np.where((label == 1) | (label == 2)))
    fluid = np.array(np.where(label == 2))

    if(len(np.where(label == 2)[0]) == 0):
        # 액체 없는 경우 컵 밑면 == 컵 밑면 근데 액체 없으면 일정 비율만큼 들어왔는지 검사할 필요가 X,,
        # cup_bottom_height = cup[0].max()
        # cup_bottom = cup[1, np.where(cup[0] == cup_bottom_height)[0]]
        return label, False, "액체가 인식되지 않았습니다", "no"
    else:
        # 액체 있는 경우 컵 밑면 == 액체 밑면
        cup_bottom_height = fluid[0].max()

    cup_top_height = cup[0].min()
    fluid_top_height = fluid[0].min()

    _, _, valid_height = calculateVolumnByPart(
        cup, fluid, cup_top_height, cup_bottom_height, fluid_top_height, ratio, 5)

    # print('valid_cup_volumn: ', valid_cup_volumn)
    # print('fluid_volumn: ', fluid_volumn)
    # print(fluid_volumn/valid_cup_volumn*100)

    print("valid_fluid_row, fluid_top_height : ",
          valid_height, fluid_top_height)

    # 어플 화면 상 선 출력 위치 [기준 row, 왼쪽 col, 오른쪽 col]
    lineLoc = [valid_height, cup[1].min(), cup[1].max()]
    lineLoc = [int(x) for x in lineLoc]

    label[int(valid_height)-3: int(valid_height) +
          3, cup[1].min():cup[1].max()] = 3

    if((fluid_top_height <= valid_height+4) and (fluid_top_height >= valid_height-4)):
        return lineLoc, True, '적정량을 따랐습니다. :) 잠시 기다려주세요', "good"
    elif (fluid_top_height > valid_height+4):
        return lineLoc, False, '재료를 더 따라주세요', "under"
    else:
        return lineLoc, False, '적정량을 초과하였습니다. 적정선에 맞도록 재료를 덜어주세요', "over"


def calculateVolumnByPart(cup, fluid, cup_h_top, cup_h_bottom, fluid_h_top, ratio, part_num):
    # 컵/액체 segmentaion, 컵 최상단/최하단 높이, 액체 최상단 높이, 비율, 부피 나눌 개수.
    volumn_sum, fluid_volumn_sum = 0, 0
    h_part_bottom, part_height = cup_h_bottom, int(
        (cup_h_bottom-cup_h_top)/part_num)
    fluid_volumn_sum = 0

    for i in range(part_num):
        h_part_top = h_part_bottom-part_height
        # 컵을 나눈 부분 중 가장 위의 부분에 포함된 경우 top을 컵 최상단 높이로 지정
        if((h_part_top < cup_h_top+part_height) and (h_part_top > cup_h_top)):
            h_part_top = cup_h_top
        part_top = cup[1, np.where(cup[0] == h_part_top)[0]]
        part_top_radius = int((part_top.max() - part_top.min())/2)
        part_bottom = cup[1, np.where(cup[0] == h_part_bottom)[0]]
        part_bottom_radius = int((part_bottom.max() - part_bottom.min())/2)

        # 현재 값을 구하는 중인 부분에 액체가 포함된 경우
        if((fluid_h_top >= h_part_top) and (fluid_h_top <= h_part_bottom)):
            fluid_top = fluid[1, np.where(fluid[0] == fluid_h_top)[0]]
            fluid_top_radius = int((fluid_top.max() - fluid_top.min())/2)
            if(part_bottom_radius < fluid_top_radius):
                short_radius = part_bottom_radius
                long_radius = fluid_top_radius
            else:
                short_radius = fluid_top_radius
                long_radius = part_bottom_radius

            if(long_radius != short_radius):
                virtual_height_part_ = (h_part_bottom-fluid_h_top)*short_radius / \
                    (long_radius - short_radius)

                v1_ = 3.14*long_radius*long_radius * \
                    (h_part_bottom-fluid_h_top+virtual_height_part_)/3
                v2_ = 3.14*short_radius * \
                    short_radius*(virtual_height_part_)/3
                fluid_volumn_sum = volumn_sum + (v1_-v2_)
            else:
                fluid_volumn_sum = volumn_sum + 3.14 * long_radius * \
                    long_radius * (h_part_bottom - fluid_h_top)

        # 역사다리꼴이 아닌 사다리꼴 형태를 띄는 경우 두 반지름 값을 바꿔 계산할 필요 O
        if(part_bottom_radius > part_top_radius):
            short_radius = part_top_radius
            long_radius = part_bottom_radius
        else:
            short_radius = part_bottom_radius
            long_radius = part_top_radius

        if(long_radius != short_radius):
            virtual_height_part = (h_part_bottom-h_part_top)*short_radius / \
                (long_radius - short_radius)

            v1 = 3.14*long_radius*long_radius * \
                (h_part_bottom-h_part_top+virtual_height_part)/3
            v2 = 3.14*short_radius*short_radius*(virtual_height_part)/3

            volumn_sum = volumn_sum + (v1-v2)
        else:
            volumn_sum = volumn_sum + 3.14 * long_radius * \
                long_radius * (h_part_bottom-h_part_top)

        h_part_bottom = h_part_bottom - part_height

    valid_volumn = volumn_sum * 0.8 * ratio/100

    h_part_bottom = cup_h_bottom
    virtual_volumn_sum = 0
    for i in range(part_num):
        h_part_top = h_part_bottom-part_height
        # 컵을 나눈 부분 중 가장 위의 부분에 포함된 경우 top을 컵 최상단 높이로 지정
        if((h_part_top < cup_h_top+part_height) and (h_part_top > cup_h_top)):
            h_part_top = cup_h_top

        part_top = cup[1, np.where(cup[0] == h_part_top)[0]]
        part_top_radius = int((part_top.max() - part_top.min())/2)
        part_bottom = cup[1, np.where(cup[0] == h_part_bottom)[0]]
        part_bottom_radius = int((part_bottom.max() - part_bottom.min())/2)

        # 역사다리꼴이 아닌 사다리꼴 형태를 띄는 경우 두 반지름 값을 바꿔 계산할 필요 O
        if(part_bottom_radius > part_top_radius):
            short_radius = part_top_radius
            long_radius = part_bottom_radius
        else:
            short_radius = part_bottom_radius
            long_radius = part_top_radius

        if(long_radius != short_radius):
            virtual_height_part = (h_part_bottom-h_part_top)*short_radius / \
                (long_radius - short_radius)

            v1 = 3.14*long_radius*long_radius * \
                (h_part_bottom-h_part_top+virtual_height_part)/3
            v2 = 3.14*short_radius*short_radius*(virtual_height_part)/3

            virtual_volumn_sum = virtual_volumn_sum + (v1-v2)
        else:
            virtual_volumn_sum = virtual_volumn_sum + 3.14 * \
                long_radius * long_radius * (h_part_bottom-h_part_top)

        height = 0
        if (virtual_volumn_sum > valid_volumn):
            part_volumn = valid_volumn - (virtual_volumn_sum - (v1-v2))
            height = h_part_bottom - \
                calcalateValidHeight(
                    part_volumn, part_bottom_radius, virtual_height_part)
            break

        h_part_bottom = h_part_bottom - part_height

    return volumn_sum, fluid_volumn_sum, height


def calcalateValidHeight(volumn, bottom_radius, virtual_height):

    radius_3 = 3*bottom_radius * \
        (volumn + 3.14*bottom_radius*bottom_radius *
         virtual_height/3)/(3.14*virtual_height)
    radius = radius_3**(1.0/3.0)
    height = virtual_height*radius/bottom_radius - virtual_height

    return height


def trimLabel(image_name, seg_map):
    # 이미지 읽기
    img = cv2.imread(image_name)
    img = cv2.resize(img, dsize=(513, 513),
                     interpolation=cv2.INTER_AREA)
    # 선명도 올리는데 이용
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
    # 이미지 축소 -> 선명도 -> canny -> 확대
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small_dst = cv2.resize(gray, dsize=(128, 128),
                           interpolation=cv2.INTER_AREA)
    small_dst = clahe.apply(small_dst)
    small_canny = auto_canny(small_dst)
    small_canny_enlarge = cv2.resize(small_canny, dsize=(513, 513),
                                     interpolation=cv2.INTER_AREA)

    # 라벨 읽기
    label_012 = seg_map  # readLabel(image_name)
    label = arr2img(label_012)
    label_gray = cv2.cvtColor(label*120, cv2.COLOR_RGB2GRAY)

    # cup만 segmentation하도록 label값 조정
    # cup: 70, liquid: 14, background: 36
    # 액체가 무조건 컵 안에 담겨있을 것으로 가정함
    if(np.count_nonzero(label_gray == 14) == 0):
        # 액체 없는 경우 --> 기존에는 진행을 하지 않았으나, 배경 인식을 위해 필요해짐.
        # 컵 부분만 mask해 배경에 수평edge 있는지 확인.
        label_cup = label_gray
    else:
        # 액체 있는 경우(!0) # 액체일 경우 cup label으로 변환, 아닐 경우 label 그대로
        label_cup = np.where(label_gray == 14, 70, label_gray)

    # cup segmentation canny -> contour
    label_cup_canny = auto_canny(label_cup)
    contours_cup, _ = cv2.findContours(
        label_cup_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  # 첫번째 차원 : 각 윤곽선 , 두번째 : 각 점 , 3 : 무조건 [0],4[0,1] : x[0]y[1] / contours[0][0] =  [[116  11]]
    # cup 개수 확인 후 2개 이상일 경우 false return
    # label 변경 시(컵2개이상인경우) return. 컵 여러 개인 경우 segmentation 또한 일그러지기 때문에 추가 trim 하는 대신 return 하기로 결정.
    if(len(contours_cup) >= 4):
        return False, label_012, '컵이 2개 이상 인식되었습니다'

    # 가장 긴 contour를 cup의 contour로 가정
    cnt_cup = []
    for i in contours_cup:
        if (len(i) >= len(cnt_cup)):
            cnt_cup = i

    # 컵 label이 뾰족해 이미지 상단과(액체때문) 하단에(가까워서) 닿은 경우
    # cup edge인 cun_cup이 좌,우로 쪼개지므로 해당 상황에서의 후처리 방지.(안 할 경우 액체label 사라짐.)
    cup_left, cup_right, cup_top, cup_bottom = findMostPoint(cnt_cup)
    cup_top = np.array(cup_top)
    if(cup_top[1] == 0 and cup_bottom[1] == 512):
        return False, label_012, '컵을 멀리서 촬영해주세요'

    # 컵에 액체가 많이 따라진 경우는 다루지 않을 확률이 높으며 윗면 edge가 많이 남으면 결과에 부정적 영향을 줌
    # 컵의 위에서 아래로 내려가며 양 끝점의 거리 차를 비교했을 때 해당 값이 작아질 경우가 윗면의 지름이라 간주
    # upper_diameter: 윗면의 지름, upper_diameter_idx: 윗면 지름의 높이값 idx
    upper_diameter, upper_diameter_idx = 0, 0
    cnt_cup_copy = cnt_cup.reshape(len(cnt_cup), 2)
    count, delete = 0, False
    for idx in range(cup_top[1], int(cup_bottom[1]/4 + cup_top[1]/4*3)+1):  # 컵 상단에서 아래로 25%
        cup_row = cnt_cup_copy[np.where(cnt_cup_copy[:, 1] == idx)]
        diameter = cup_row[:, 0].max() - cup_row[:, 0].min()
        if(diameter < 50):
            # 따라지는 액체로 인해 컵이 위로 뾰족하게 인식되는 경우.
            label_012[idx, :] = 0
            delete = True
        elif(diameter-upper_diameter > 2):
            upper_diameter = diameter
            upper_diameter_idx = idx
            count = 0
            if(delete):
                cup_top[1] = upper_diameter_idx
                delete = False
        else:
            count += 1
            if(count >= 5):
                break

    # 컵 위에서 촬영하는 경우 제한. (컵 윗면의 세로반지름/가로반지름이 긴 경우)
    cup_upper_height = upper_diameter_idx - cup_top[1]

    print('cup_upper_height: ', cup_upper_height)
    print(cup_upper_height / upper_diameter)
    if(cup_upper_height / upper_diameter > 0.25):
        # 컵 윗면이 많이 나온 경우
        return False, label_012, '컵의 정면을 촬영해주세요'

    # label_cup에서 컵이 있는 부분만 True, 나머지는 false로 mask 생성
    label_cup_mask = label_cup == 70
    # mask의 컵 부분(True/False) 축소/확대시키기
    reduce_label_cup_mask = modifyMask(label_cup_mask, -10)

    # 우선적으로 상단 15% 지움 이후 액체 양 검사해 적을 경우 더 지울 것.
    limit = int(cup_top[1]/20*17 + cup_bottom[1]/20*3)
    reduce_label_cup_mask[:limit, :] = False

    # cup: 70, liquid: 14, background: 36
    if(np.count_nonzero(label_gray == 14) == 0):
        # 액체 없는 경우 --> 기존에는 진행을 하지 않았으나, 배경 인식을 위해 필요해짐.
        # 컵 부분만 mask해 배경에 수평edge 있는지 확인.
        # small_canny_cup_enlarge 컵이 있는 부분만 남기고 나머지는 지움. 이 때 1차원 되므로 reshape해준다.
        small_dst_cup = cv2.resize(gray, dsize=(64, 64),
                                   interpolation=cv2.INTER_AREA)
        small_dst_cup = clahe.apply(small_dst_cup)
        small_canny_cup = auto_canny(small_dst_cup)
        reduce_label_cup_mask_f = reduce_label_cup_mask.astype(np.uint8)
        reduce_label_cup_mask_f[:int(
            cup_top[1]/100*73 + cup_bottom[1]/100*27), :] = 0
        reduce_label_cup_mask_f[int(
            cup_top[1]/100*27 + cup_bottom[1]/100*73):, :] = 0
        reduce_label_cup_mask_f = cv2.resize(reduce_label_cup_mask_f, dsize=(64, 64),
                                             interpolation=cv2.INTER_AREA)
        filtered_canny_cup = np.where(
            reduce_label_cup_mask_f, small_canny_cup, 0)
        filtered_canny_cup = np.reshape(filtered_canny_cup, (64, 64))

        i = 0
        for row_canny, row_mask in zip(filtered_canny_cup, reduce_label_cup_mask_f):
            count_mask = sum(row_mask == 1)/2
            count_canny = sum(row_canny == 255)
            i += 1
            if(sum(row_mask == 1) != 0 and (count_mask <= count_canny)):
                # canny와 mask 행 별로 값 있는 픽셀 수 계산해
                # print('컵 표면을 닦거나 뒷 배경에 수평선이 보이지 않도록 해주세요.')
                # print(count_mask, count_canny)
                return False, label_012, '컵 표면을 닦거나 뒷 배경에 수평선이 보이지 않도록 해주세요'
            else:
                continue
        return False, label_012, '액체가 인식되지 않았습니다'

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
        if (len(i) >= len(cnt_fluid)):  # TODO : max 로 찾을 수 있지 않을까
            cnt_fluid = i

    # 액체의 극점, middle_top/bottom 찾기 (이 때, middle_top[1] < middle_bottom[1])
    fluid_left, fluid_right, _, _ = findMostPoint(cnt_fluid)
    fluid_middle_top, _ = findMiddlePoint(
        cnt_fluid, int((fluid_left[0]+fluid_right[0])/2))
    fluid_middle_top = np.array(fluid_middle_top)

    fluid = np.array(np.where(label_012 == 2))
    fluid_top_height = fluid[0].min()
    idx = np.where(fluid[0] == fluid_top_height)

    if len(idx[0]) == 1:
        fluid_middle_top[0], fluid_middle_top[1] = int(
            (cup_left[0]+cup_right[0])/2), fluid[0][idx]
    else:
        fluid_middle_top[0], fluid_middle_top[1] = int(
            (cup_left[0]+cup_right[0])/2), fluid[0][idx[0][0]]

    if(fluid_middle_top[1] >= int(cup_top[1]/2 + cup_bottom[1]/2)):
        # 컵 윗면이 많이 나오지 X (정면에 가까움) + 컵의 반 이하로 따라진 경우 상단 25% 지움
        limit = int(cup_top[1]/4*3 + cup_bottom[1]/4)

    reduce_label_cup_mask[:limit, :] = False
    # 컵 바닥부터 액체 높이의 2/3 지움 -> 수면 주변의 주요 edge만 남기기 위함
    reduce_label_cup_mask[int(
        fluid_middle_top[1]/3*1 + cup_bottom[1]/3*2):cup_bottom[1], :] = False

    # small_canny_enlarge에서 컵이 있는 부분만 남기고 나머지는 지움. 이 때 1차원 되므로 reshape해준다.
    filtered_canny = np.where(
        reduce_label_cup_mask, small_canny_enlarge, 0)
    filtered_canny = np.reshape(filtered_canny, (513, 513))

    # 컵부분만 남긴 edge들을 contour로 변환. 각 edge의 길이 비교를 위함.
    contours_filtered_cup, _ = cv2.findContours(
        filtered_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    if(len(contours_filtered_cup) < FLUID_APPROX_EDGE_NUM):
        # 비교 가능한 액체의 edge가 일정 개수가 되지 않을 경우 error 코드 return
        return False, label_012, '액체 감지에 오류가 발생했습니다'

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

    # 길이가 가장 긴(면적이큰) 일정 개수의 edge 중심점을 계산해 M에 저장
    M = np.empty((0, 3), dtype=int)
    for i in range(FLUID_APPROX_EDGE_NUM):
        M1 = cv2.moments(contours_filtered_cup[int(
            contours_filtered_cup_area_sort[-1*(i+1)][0])])
        x = int(M1['m10']/M1['m00'])
        y = int(M1['m01']/M1['m00'])
        M = np.append(M, np.array([[-1*(i+1), x, y]]), axis=0)

    # 액체의 middle_top과 가장 가까운 중심점을 갖는 edge 찾기
    tmp = []
    for i in range(FLUID_APPROX_EDGE_NUM):
        tmp = np.append(tmp, get_dist(M[i][1:], fluid_middle_top))

    approx_line = contours_filtered_cup[int(
        contours_filtered_cup_area_sort[M[tmp.argmin()][0]][0])]

    # approx_line의 중간 10점 정도의 평균점 찾아 label 수정
    # + 컵 상단 cup_upper_height만큼 지우기, 액체 상단은 더하기
    label_012 = trimFluid(approx_line, fluid_middle_top,
                          label_012, cup_upper_height)
    # 액체 label을 컵 따라 매끈하게 만드는 함수
    # + 액체 밑면 중심을 통과하도록 평평하게 만들며 컵의 밑면도 지움
    label_012 = trimFluidFollowCup(label_012, cnt_cup)

    return True, label_012, 'good'
