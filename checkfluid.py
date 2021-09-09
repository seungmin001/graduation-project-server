import numpy as np

def checkAreaOfLiquid(label, ratio):
    # 컵과 액체의 면적을 파악하고 액체의 양이 원하는 비율만큼 채워졌는지 검사
    _, label_count = np.unique(label, return_counts=True)
    print(label_count)
    cup_area = label_count[1] + label_count[2]
    liquid_area = label_count[2]
    valid_cup_area = cup_area*0.8

    print(liquid_area/valid_cup_area*100)
    #####################
    cup = np.array(np.where((label == 1) | (label == 2))) # array(행) array(열)
    fluid = np.array(np.where(label == 2))

    if(len(fluid[0]) == 0):
        # 액체 없는 경우 컵 밑면 == 컵 밑면 근데 액체 없으면 일정 비율만큼 들어왔는지 검사할 필요가 X,,
        # cup_bottom_height = cup[0].max()
        # cup_bottom = cup[1, np.where(cup[0] == cup_bottom_height)[0]]
        return False
    else:
        # 액체 있는 경우 컵 밑면 == 액체 밑면
        cup_bottom_height = fluid[0].max()
        cup_bottom = fluid[1, np.where(fluid[0] == cup_bottom_height)[0]]

    cup_top_height = cup[0].min() # (cup, fluid 인식 부분 중) 가장 작은 행 인덱스
    # cup_top = cup[1, np.where(cup[0] == cup_top_height)[0]] # 컵 맨위 col 들 arr

    cup_height = cup_bottom_height - cup_top_height # 실제 cup height

    required_fluid_row=cup_bottom_height-int((cup_height)*0.8 * (ratio/100)) # 원하는 비율
    print("required_fluid_row : ",required_fluid_row)
    print(cup[1].max(),cup[1].min())
    label[int(required_fluid_row) -
          1:int(required_fluid_row)+1, cup[1].min():cup[1].max()] = 3
    ######################
    # if((ratio-2 <= liquid_area/valid_cup_area*100) & (ratio+2 >= liquid_area/valid_cup_area*100)):
    if((ratio-2 <= liquid_area/valid_cup_area*100)): # 일단 ratio 기준 넘으면 True 처리    
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
        return False
    else:
        # 액체 있는 경우 컵 밑면 == 액체 밑면
        cup_bottom_height = fluid[0].max()
        cup_bottom = fluid[1, np.where(fluid[0] == cup_bottom_height)[0]]

    cup_top_height = cup[0].min() # (cup, fluid 인식 부분 중) 가장 작은 행 인덱스
    cup_top = cup[1, np.where(cup[0] == cup_top_height)[0]] # 컵 맨위 col 들 arr

    cup_top_radius = int((cup_top.max() - cup_top.min())/2)
    cup_bottom_radius = int((cup_bottom.max() - cup_bottom.min())/2)
    cup_height = cup_bottom_height - cup_top_height # 실제 cup height

    fluid_top_height = fluid[0].min()
    fluid_top = fluid[1, np.where(fluid[0] == fluid_top_height)[0]]
    fluid_top_radius = int((fluid_top.max() - fluid_top.min())/2)
    fluid_height = cup_bottom_height - fluid_top_height

    # label[cup_top_height, cup_top] = 250
    # label[cup_bottom_height, cup_bottom] = 250
    # label[fluid_top_height, fluid_top] = 150

    # cv2.imshow('', label)
    # cv2.waitKey(0)

    cup_volumn, fluid_volumn, valid_height = calculateVolumnByPart(
        cup, fluid, cup_top_height, cup_bottom_height, fluid_top_height, ratio, 5)
    # fluid_volumn = calculateVolumnByPart(
    #     fluid, fluid_top_height, cup_bottom_height, 5)

    valid_cup_volumn = cup_volumn*0.8
    #print('valid_cup_volumn: ', valid_cup_volumn)
    #print('fluid_volumn: ', fluid_volumn)
    print(fluid_volumn/valid_cup_volumn*100)
    # valid_height = calcalateValidHeight(
    #     valid_cup_volumn, ratio, cup_bottom_radius, virtual_height)
    required_fluid_row=cup_bottom_height-int((cup_height) * (ratio/100)) # 원하는 비율
    print("required_fluid_row : ",required_fluid_row)
    label[int(required_fluid_row) -
          1:int(required_fluid_row)+1, cup_top.min():cup_top.max()] = 3

    if((ratio-2 <= fluid_volumn/valid_cup_volumn*100)): # & (ratio+2 >= fluid_volumn/valid_cup_volumn*100)):
        return True
    else:
        return False


def calculateVolumnByPart(segment, h_top, h_bottom, part_num):
    volumn_sum = 0
    h_part_top, part_height = h_top, int((h_bottom-h_top)/part_num)
    for i in range(part_num):
        h_part_bottom = h_part_top+part_height
        if((h_part_bottom > h_bottom-part_height) & (h_part_bottom < h_bottom)):
            h_part_bottom = h_bottom

        part_top = segment[1, np.where(segment[0] == h_part_top)[0]]
        part_top_radius = int((part_top.max() - part_top.min())/2)
        part_bottom = segment[1, np.where(segment[0] == h_part_bottom)[0]]
        part_bottom_radius = int((part_bottom.max() - part_bottom.min())/2)

        virtual_height_part = (h_part_bottom-h_part_top)*part_bottom_radius / \
            (part_top_radius - part_bottom_radius)

        v1 = 3.14*part_top_radius*part_top_radius * \
            (h_part_bottom-h_part_top+virtual_height_part)/3
        v2 = 3.14*part_bottom_radius*part_bottom_radius*(virtual_height_part)/3

        volumn_sum = volumn_sum + (v1-v2)

        h_part_top = h_part_top + part_height
        # print(volumn_sum)

    return volumn_sum

def calculateVolumnByPart(cup, fluid, cup_h_top, cup_h_bottom, fluid_h_top, ratio, part_num):
    volumn_sum, fluid_volumn_sum = 0, 0
    h_part_bottom, part_height = cup_h_bottom, int(
        (cup_h_bottom-cup_h_top)/part_num)

    # for i in range(part_num):
    #     h_part_top = h_part_bottom-part_height
    #     if((h_part_top < cup_h_top+part_height) & (h_part_top > cup_h_top)):
    #         h_part_top = cup_h_top
    #     print(h_part_top, h_part_bottom)
    #     h_part_bottom = h_part_bottom - part_height

    for i in range(part_num):
        h_part_top = h_part_bottom-part_height
        if((h_part_top < cup_h_top+part_height) & (h_part_top > cup_h_top)):
            h_part_top = cup_h_top
        part_top = cup[1, np.where(cup[0] == h_part_top)[0]]
        part_top_radius = int((part_top.max() - part_top.min())/2)
        part_bottom = cup[1, np.where(cup[0] == h_part_bottom)[0]]
        part_bottom_radius = int((part_bottom.max() - part_bottom.min())/2)

        virtual_height_part = (h_part_bottom-h_part_top)*part_bottom_radius / \
            (part_top_radius - part_bottom_radius)

        v1 = 3.14*part_top_radius*part_top_radius * \
            (h_part_bottom-h_part_top+virtual_height_part)/3
        v2 = 3.14*part_bottom_radius*part_bottom_radius*(virtual_height_part)/3

        if((fluid_h_top >= h_part_top) & (fluid_h_top <= h_part_bottom)):
            fluid_top = fluid[1, np.where(fluid[0] == fluid_h_top)[0]]
            fluid_top_radius = int((fluid_top.max() - fluid_top.min())/2)
            virtual_height_part_ = (h_part_bottom-fluid_h_top)*part_bottom_radius / \
                (fluid_top_radius - part_bottom_radius)

            v1_ = 3.14*fluid_top_radius*fluid_top_radius * \
                (h_part_bottom-fluid_h_top+virtual_height_part_)/3
            v2_ = 3.14*part_bottom_radius * \
                part_bottom_radius*(virtual_height_part_)/3
            fluid_volumn_sum = volumn_sum + (v1_-v2_)

        volumn_sum = volumn_sum + (v1-v2)

        h_part_bottom = h_part_bottom - part_height
        # print(volumn_sum)

    valid_volumn = volumn_sum * 0.8 * ratio/100
    # print('valid_volumn:', valid_volumn)
    virtual_volumn_sum = 0
    for i in range(part_num):
        h_part_bottom = h_part_top+part_height
        if((h_part_bottom > cup_h_bottom-part_height) & (h_part_bottom < cup_h_bottom)):
            h_part_bottom = cup_h_bottom

        part_top = cup[1, np.where(cup[0] == h_part_top)[0]]
        part_top_radius = int((part_top.max() - part_top.min())/2)
        part_bottom = cup[1, np.where(cup[0] == h_part_bottom)[0]]
        part_bottom_radius = int((part_bottom.max() - part_bottom.min())/2)

        virtual_height_part = (h_part_bottom-h_part_top)*part_bottom_radius / \
            (part_top_radius - part_bottom_radius)

        v1 = 3.14*part_top_radius*part_top_radius * \
            (h_part_bottom-h_part_top+virtual_height_part)/3
        v2 = 3.14*part_bottom_radius*part_bottom_radius*(virtual_height_part)/3

        virtual_volumn_sum = virtual_volumn_sum + (v1-v2)
        # print(virtual_volumn_sum)
        if (virtual_volumn_sum > valid_volumn):
            part_volumn = valid_volumn - (virtual_volumn_sum - (v1-v2))
            height = h_part_bottom - \
                calcalateValidHeight(
                    part_volumn, part_bottom_radius, virtual_height_part)
            break

        part_top = part_top + part_height

    return volumn_sum, fluid_volumn_sum, height

def calculateVolumn(r_top, r_bottom, height, virtual_height):
    v1 = 3.14*r_top*r_top*(height+virtual_height)/3
    v2 = 3.14*r_bottom*r_bottom*(virtual_height)/3
    return v1-v2


def calcalateValidHeight(volumn, bottom_radius, virtual_height):

    radius_3 = 3*bottom_radius * \
        (volumn + 3.14*bottom_radius*bottom_radius *
         virtual_height/3)/(3.14*virtual_height)
    radius = radius_3**(1.0/3.0)
    height = virtual_height*radius/bottom_radius - virtual_height

    return height
