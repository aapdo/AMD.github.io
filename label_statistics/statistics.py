import scipy.io
import pandas as pd
import numpy as np
import mat4py
import re
import os
import csv

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

def get_correct_origin_diff(xlsx_file_path):
    '''
    기존 라벨링 데이터와 수정한 라벨링 데이터 사이에서 얼마나 수정되었는지 계산하는 함수
    '''
    df = pd.read_excel(xlsx_file_path)
    
    for index, row in df.iterrows():
        print(1)


    return df


if __name__ == '__main__':
    mat_file_path = "./market_attribute.mat"
    correct_label_file_path = "./correct_label.xlsx"
    img_file_path = "./labelTarget"

    # 정규 표현식
    

    # 정규 표현식에 매칭되는 부분 찾기

    unique_file_names = get_all_files_in_folder(img_file_path)
    '''
    with open("./img_names.csv", 'w', newline='') as file:
        writer = csv.writer(file)
        for name in unique_file_names:
            writer.writerow([name])
    '''

    load_xlsx_label(correct_label_file_path)

        

    mat_attribtue_train, mat_attribtue_test, train_indexes, test_indexes = load_mat_attr(mat_file_path)