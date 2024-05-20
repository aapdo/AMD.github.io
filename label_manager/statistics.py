import scipy.io
import pandas as pd
import numpy as np
import mat4py
import re
import os
import csv
# import matplot as plt
import matplotlib.pyplot as plt

root_path = "/Users/imjun-yeong/googleDrive_/dgu/3_1/개별연구/csid_git/label_manager"
root_path = '/root/amd/label_manager'
dataset_path = root_path + "/dataset/Market-1501-v15.09.15"
train_data_path = dataset_path + "/bounding_box_train"
test_data_path = dataset_path + "/bounding_box_test"
query_data_path = dataset_path + "/query"
mat_file_path = root_path + "/market_attribute.mat"
correct_label_file_path = root_path + "/for_statistics.xlsx"
img_file_path = root_path + "/labelTarget"

down_attr_list = ['down-black', 'down-blue', 'down-brown', 'down-gray', 'down-green', 'down-pink', 'down-purple', 'down-white', 'down-yellow', 'down']
upper_attr_list = ['up-black', 'up-blue', 'up-green', 'up-gray', 'up-purple', 'up-red', 'up-white', 'up-yellow', 'up']

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
    pattern = r'^.{4}'
    # 전체 라벨 수
    total_label_number = 0
    # 하나라도 라벨이 수정된 사진의 수
    cnt_all_attr_diff = 0
    # 각 속성별로 수정된 수
    cnt_upper_diff = 0
    cnt_lower_diff = 0

    for index, row in correct_label_df.iterrows():
        total_label_number += 1

        img_name = row["origin_img_name"]
        # 현재 확인하는 이미지의 식별 번호
        pid = re.match(pattern, img_name).group()
        # 기존 라벨 속성
        original_attr = mat_attribute.loc[pid]
        # 수정이 한번이라도 이루어졌는지 체크하는 변수
        
        total_flag = False
        down_flag = False
        up_flag = False
         
        for key, value in original_attr.items():    
            if value != row[key]:
                if key in down_attr_list:
                    down_flag = True
                if key in upper_attr_list:
                    up_flag = True
                total_flag = True
        if total_flag:
            cnt_all_attr_diff += 1
        if up_flag:
            cnt_upper_diff += 1
        if down_flag:
            cnt_lower_diff += 1
    return total_label_number, cnt_all_attr_diff, cnt_upper_diff, cnt_lower_diff

def draw_pie_chart(labels, sizes, title, chart_name):
    fig, ax = plt.subplots()
    # autopct: 조각의 비율
    # startangle: 시작하는 각도
    # shadow: 그림자
    # explode: 특정 조각을 돌출
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=False)
    ax.axis('equal')  # 원형을 유지하기 위해 설정
    plt.title(title)

    plt.savefig(root_path + "/statistics/" + chart_name + ".png")
    plt.close(fig)

def draw_modified_ratio_pie_chart(total_label_number, cnt_all_attr_diff, cnt_upper_diff, cnt_lower_diff):
    '''
    1. 전체 라벨 중 수정한 비율이 얼마나 되는지
    2. 수정 후 속성 분포가 어떻게 되는지
    3. 각 속성별로 수정된 비율이 얼마나 되는지
    '''

    # 전체에서 수정된 비율
    labels = ['Unmodified Attributes', 'Modified Attributes']

    title = "Ratio of Attributes Modified by One or More"
    chart_name = "all_modification_ratio"
    modified_ratio = cnt_all_attr_diff / total_label_number * 100
    sizes = [100-modified_ratio, modified_ratio]
    draw_pie_chart(labels=labels, sizes=sizes, title=title, chart_name=chart_name)

    # 상의가 수정된 비율
    title = "Ratio of Modified Upper Attributes"
    chart_name = "upper_modification_ratio"
    modified_ratio = cnt_upper_diff / total_label_number * 100
    sizes = [100-modified_ratio, modified_ratio]
    draw_pie_chart(labels=labels, sizes=sizes, title=title, chart_name=chart_name)

    # 하의가 수정된 비율
    title = "Ratio of Modified Lower Attributes"
    chart_name = "lower_modification_ratio"
    modified_ratio = cnt_lower_diff / total_label_number * 100
    sizes = [100-modified_ratio, modified_ratio]
    draw_pie_chart(labels=labels, sizes=sizes, title=title, chart_name=chart_name)

def draw_bar_chat_attribute_distribution(total_label_number, correct_label_df):
    cnt_correct_attr = {
        'backpack': 0,
        'bag': 0,
        'handbag': 0,
        'down-black': 0,
        'down-blue': 0,
        'down-brown': 0,
        'down-gray': 0,
        'down-green': 0,
        'down-pink': 0,
        'down-purple': 0,
        'down-white': 0,
        'down-yellow': 0,
        'up-black': 0,
        'up-blue': 0,
        'up-green': 0,
        'up-gray': 0,
        'up-purple': 0,
        'up-red': 0,
        'up-white': 0,
        'up-yellow': 0,
        'hat': 0,

        'gender-male': 0,
        'long-uppper': 0,
        'long-hair': 0,    
    }
    '''
        머리 긴거로 세기
        윗옷 긴거로 세기
        성별 남자로 세기
        down, age, clothes 없앰

        1, 2번이 의미가 다른거:
            up: 1번이면 카운트
            hair: 2번이면 카운트
            gender: 1번이면 카운트

    '''
    subset_df = correct_label_df[['up', 'hair', 'gender']]
    correct_label_df.drop(['age', 'clothes', 'down', 'up', 'hair', 'gender', 'origin_img_name'], axis=1, inplace=True)

    # DataFrame에서 각 속성의 개수 세기
    #for index, row in correct_label_df.iterrows():
    for col in correct_label_df.columns:
        count_2 = correct_label_df[col].value_counts()[2]
        cnt_correct_attr[col] = (count_2 / total_label_number)
        # if correct_label_df
    for col in subset_df:
        if col == "up":
            count_1 = subset_df[col].value_counts()[1]
            cnt_correct_attr['long-uppper'] = (count_1 / total_label_number)
        elif col == "gender":
            count_1 = subset_df[col].value_counts()[1]
            cnt_correct_attr['long-hair'] = (count_1 / total_label_number)
        else:
            count_2 = subset_df[col].value_counts()[2]
            cnt_correct_attr['gender-male'] = (count_2 / total_label_number)

    # 막대 그래프 그리기
    fig, ax = plt.subplots(figsize=(10, 5))
    categories = list(cnt_correct_attr.keys())
    values = list(cnt_correct_attr.values())

    # 수평 막대 그래프 그리기, 색상 및 너비 조정
    ax.barh(categories, values, color='skyblue', height=0.5)  # 높이 조정

    # 제목 및 라벨 추가
    plt.title('Statistics of Corrected Attributes in Market1501')
    plt.xlabel('Ratios in All Samples')
    plt.ylabel('Attributes')

    # x축의 범위 설정
    plt.xlim(0, 1)

    # 그래프 표시
    plt.savefig(root_path + "/statistics/all_attr_ratio_bar.png")
    plt.close(fig)

if __name__ == '__main__':


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

    # origin_img_name을 제외한 모든 열을 정수형으로 변환
    columns_to_convert = correct_label_df.columns.drop('origin_img_name')  # origin_img_name 제외
    correct_label_df[columns_to_convert] = correct_label_df[columns_to_convert].apply(pd.to_numeric, downcast='integer', errors='coerce')

    nan_in_rows = correct_label_df.isna().any(axis=1)
    nan_rows = correct_label_df[nan_in_rows]
    print("NaN이 있는 행:\n", nan_rows)

    mat_attribute_train, mat_attribute_test, train_indexes, test_indexes = load_mat_attr(mat_file_path)

    # train + test attribute
    mat_attribute = pd.concat([mat_attribute_train, mat_attribute_test], axis=0).sort_index() 
    mat_attribute = mat_attribute.drop(columns=['image_index'])

    mat_attribute.rename(columns={
        'downblack': 'down-black', 
        'downblue': 'down-blue',
        'downbrown': 'down-brown', 
        'downgray': 'down-gray', 
        'downgreen': 'down-green', 
        'downpink': 'down-pink', 
        'downpurple': 'down-purple',
        'downwhite': 'down-white', 
        'downyellow': 'down-yellow', 
        'upblack': 'up-black', 
        'upblue': 'up-blue', 
        'upgreen': 'up-green', 
        'upgray': 'up-gray',
        'uppurple': 'up-purple', 
        'upred': 'up-red', 
        'upwhite': 'up-white', 
        'upyellow': 'up-yellow',
    }, inplace=True)

    correct_label_df.rename(columns={
        'downblack': 'down-black', 
        'downblue': 'down-blue',
        'downbrown': 'down-brown', 
        'downgray': 'down-gray', 
        'downgreen': 'down-green', 
        'downpink': 'down-pink', 
        'downpurple': 'down-purple',
        'downwhite': 'down-white', 
        'downyellow': 'down-yellow', 
        'upblack': 'up-black', 
        'upblue': 'up-blue', 
        'upgreen': 'up-green', 
        'upgray': 'up-gray',
        'uppurple': 'up-purple', 
        'upred': 'up-red', 
        'upwhite': 'up-white', 
        'upyellow': 'up-yellow',
    }, inplace=True)

    total_label_number, cnt_all_attr_diff, cnt_upper_diff, cnt_lower_diff = get_correct_origin_diff(correct_label_df, mat_attribute)

    draw_modified_ratio_pie_chart(total_label_number, cnt_all_attr_diff, cnt_upper_diff, cnt_lower_diff)

    print("전체 검토한 사진: ", total_label_number)
    print("수정한 사진의 수: ", cnt_all_attr_diff)

    draw_bar_chat_attribute_distribution(total_label_number, correct_label_df)
