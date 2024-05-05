import scipy.io
import pandas as pd
import numpy as np
import mat4py
import re
import os
import csv
import matplot

cur_path = "./label_statistics"


def get_all_files_in_folder(folder_path):
    """
    라벨 변경 대상 이미지 폴더 내부에 있는 모든 파일의 이름을 가져옴
    0000_c0s0_000000_01.jpg 에서 맨 마지막 _01.jpg를 제거함
    중복이 없도록 처리해서 리턴
    
    :param folder_path: 가져올 파일이 있는 폴더의 경로
    :return: 폴더 내부에 있는 모든 파일 이름을 담은 리스트
    """
    # 폴더 내부에 있는 모든 파일 이름을 저장할 리스트
    file_names = []
    pattern = r'\d{4}_c\d+s\d+_\d{6}'
    unique_file_names = set()

    # 폴더 내부의 모든 파일과 디렉터리를 가져와서 반복
    for root, dirs, files in os.walk(folder_path):
        # 파일 이름을 리스트에 추가
        for file in files:
            matches = re.findall(pattern, file)  
            file_names.append(matches)
    for name_list in file_names:
        for name in name_list:
            unique_file_names.add(name)

    return sorted(list(unique_file_names))


def load_mat_attr(mat_file_path):
    mat_attribute_train = mat4py.loadmat(mat_file_path)["market_attribute"]["train"]
    mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)

    mat_attribute_test = mat4py.loadmat(mat_file_path)["market_attribute"]["test"]
    mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)

    # train, test에 포함된 사진 번호들
    train_indexes = mat_attribute_train.index
    test_indexes = mat_attribute_test.index

    return mat_attribute_train, mat_attribute_test, train_indexes, test_indexes

def get_correct_origin_diff(correct_label_df, mat_attribute_df):
    '''
    기존 라벨링 데이터와 수정한 라벨링 데이터 사이에서 얼마나 수정되었는지 계산하는 함수
    '''
    #pattern = r'\d{4}' 
    pattern = r'^.{4}'
    # 전체 라벨 수
    total_label_number = 0
    # 하나라도 라벨이 수정된 사진의 수
    correct_img_number = 0
    # 각 속성별로 수정된 수
    cnt_correct_attr = {
        'age': 0,
        'backpack': 0,
        'bag': 0,
        'handbag': 0,
        'downblack': 0,
        'downblue': 0,
        'downbrown': 0,
        'downgray': 0,
        'downgreen': 0,
        'downpink': 0,
        'downpurple': 0,
        'downwhite': 0,
        'downyellow': 0,
        'upblack': 0,
        'upblue': 0,
        'upgreen': 0,
        'upgray': 0,
        'uppurple': 0,
        'upred': 0,
        'upwhite': 0,
        'upyellow': 0,
        'clothes': 0,
        'down': 0,
        'up': 0,
        'hair': 0,
        'hat': 0,
        'gender': 0,        
    }

    for index, row in correct_label_df.iterrows():
        total_label_number += 1

        img_name = row["origin_img_name"]
        # 현재 확인하는 이미지의 식별 번호
        pid = re.match(pattern, img_name).group()
        # 기존 라벨 속성
        original_attr = mat_attribute.loc[pid]
        # 수정이 한번이라도 이루어졌는지 체크하는 변수
        flag = False

        for key, value in original_attr.items():
            if value != row[key]:
                cnt_correct_attr[key] += 1
                flag = True
        if flag:
            correct_img_number += 1

    return total_label_number, correct_img_number, cnt_correct_attr

def draw_graph(total_label_number, correct_img_number, cnt_correct_attr):
    '''
    2. 전체 라벨 중 수정한 비율이 얼마나 되는지
    3. 수정 후 속성 분포가 어떻게 되는지
    4. 각 속성별로 수정된 비율이 얼마나 되는지
    '''

    return

if __name__ == '__main__':
    mat_file_path = cur_path + "/market_attribute.mat"
    correct_label_file_path = cur_path + "/correct_label.xlsx"
    img_file_path = cur_path + "/labelTarget"

    # 정규 표현식
    

    # 정규 표현식에 매칭되는 부분 찾기

    unique_file_names = get_all_files_in_folder(img_file_path)
    '''
    with open("./img_names.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        for name in unique_file_names:
            writer.writerow([name])
    '''

    correct_label_df = pd.read_excel(correct_label_file_path)
    mat_attribute_train, mat_attribute_test, train_indexes, test_indexes = load_mat_attr(mat_file_path)

    # train + test attribute
    mat_attribute = pd.concat([mat_attribute_train, mat_attribute_test], axis=0).sort_index() 
    mat_attribute = mat_attribute.drop(columns=['image_index'])

    total_label_number, correct_img_number, cnt_correct_attr = get_correct_origin_diff(correct_label_df, mat_attribute)

    
