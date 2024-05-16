import os
import pandas as pd
import re
import csv
import copy
import shutil

root_path = "/Users/imjun-yeong/googleDrive_/dgu/3_1/개별연구/csid_git/label_statistics"
dataset_path = root_path + "/Market-1501-v15.09.15"
train_data_path = dataset_path + "/bounding_box_train"
test_data_path = dataset_path + "/bounding_box_test"
query_data_path = dataset_path + "/query"
correct_label_path = root_path + "/label_gene.xlsx"
new_dataset_path = root_path + "/Market-1586-v24.05.12"
new_train_data_path = new_dataset_path + "/bounding_box_train"
new_test_data_path = new_dataset_path + "/bounding_box_test"
new_query_data_path = new_dataset_path + "/query"

def write_dict_to_csv(file_path, output_dict: dict):
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        for key, value in output_dict.items():
            writer.writerow([key] + value)
        
def check_keys_for_jpg(dictionary):
    # 딕셔너리의 모든 키에 대해 반복
    for key in dictionary.keys():
        # 키에 '.jpg'가 포함되어 있지 않으면 False 반환
        if '.jpg' not in key:
            print(f"check_keys_for_jpg error {key} 에 jpg가 없음. ")
            exit()
            return False
    # 모든 키에 '.jpg'가 포함되어 있으면 True 반환
    return True

def remove_unlabeled_keys(attr_dict):
    # '-1' 또는 '0000'으로 시작하는 키를 제거
    keys_to_remove = [key for key in attr_dict.keys() if key.startswith('-1') or key.startswith('0000')]
    for key in keys_to_remove:
        del attr_dict[key]
    return attr_dict

def create_dict_from_filenames(directory, initial_value=None):
    """
    주어진 디렉토리 내의 모든 파일 이름으로 사전을 생성한다.
    각 키의 초기값은 `initial_value`로 설정된다.
    
    :param directory: 파일 이름을 가져올 디렉토리 경로
    :param initial_value: 모든 키의 초기 값
    :return: 파일 이름을 키로 하는 사전
    """
    # 디렉토리 내의 모든 파일과 디렉토리 목록을 가져옵니다.
    file_names = os.listdir(directory)
    file_names = [file for file in file_names if file.endswith('.jpg')] 
    # 파일 이름을 키로 하고, 초기값을 값으로 하는 사전을 생성합니다.
    # 이때 os.path.isfile을 사용하여 파일인 경우만 포함합니다.
    file_dict = {file: initial_value for file in file_names if os.path.isfile(os.path.join(directory, file))}
    
    return file_dict

def normalize_img_name(img_name):
    '''
        이미지의 이름을 정규화한다.
        정규화 규칙은 숫자4개_'c'숫자's'숫자_숫자6개 이다.
    '''
    # 정규 표현식으로 숫자4개_'c'숫자's'숫자_숫자6개 패턴 매칭
    pattern = r'\d{4}_c\d+s\d+_\d{6}'
    matches = re.findall(pattern, img_name)  
    if len(matches) >= 1:
        return matches[0]
    else:
        return ""

def find_diff_label_n_real(new_label_dict, train_attr_dict, test_attr_dict, query_attr_dict):
    '''
        수기로 라벨링을 수행했기 때문에 라벨링이 되지 않은 이미지가 존재할 수 있다.
        라벨링이 되지 않은 이미지를 찾아서 csv 파일로 출력해준다.
    '''
    # 모든 실제 데이터셋의 키 정규화
    train_keys = set(normalize_img_name(key) for key in train_attr_dict.keys())
    # print("train keys: ", train_keys)
    test_keys = set(normalize_img_name(key) for key in test_attr_dict.keys())
    # print("test keys: ", test_keys)
    query_keys = set(normalize_img_name(key) for key in query_attr_dict.keys())
    # print("query keys: ", query_keys)
    
    # 실제 데이터셋 키의 합집합
    real_data_keys = train_keys | test_keys | query_keys

    # 라벨링된 데이터의 키 정규화
    labeled_keys = set(normalize_img_name(key) for key in new_label_dict.keys())
    
    # 라벨링 데이터와 실제 데이터의 차집합을 찾아 라벨링 되었으나 실제 데이터셋에 없는 키를 찾습니다.
    # missing_in_real = labeled_keys - real_data_keys
    # print("Labels not in real data:", missing_in_real)
    missing_in_labels = real_data_keys - labeled_keys
    missing_in_labels = list(missing_in_labels)
    missing_in_labels.sort()
    print("Real data not in labels:", missing_in_labels)
    with open('./missing.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for item in missing_in_labels:
            writer.writerow([item])

def check_new_img(filtered_dict, another_dict):
    '''
    filtered_dict: 새로운 라벨에서 인물 번호를 추가해야하는 사진만 저장된 고른 dict
                    또는 사진 이름을 변경해야 하는 사진만 고른 dict
    another_dict: train, test, query가 들어옴
    
    train, test, query에 각각 어떤 이미지가 새로운 이미지로 분리되어야 하는지 리턴해줌
    '''
    # another_dict의 깊은 복사본 생성
    updated_dict = copy.deepcopy(another_dict)

    # updated_dict에서 키 목록을 리스트로 복사
    keys_in_updated = list(updated_dict.keys())
    
    # updated_dict의 각 키에 대해 조건 검사
    for key in keys_in_updated:
        if key in filtered_dict:
            # filtered_dict에 있는 경우, updated_dict의 값을 업데이트
            updated_dict[key] = filtered_dict[key]
        else:
            # filtered_dict에 없는 경우, updated_dict에서 해당 키 제거
            del updated_dict[key]

    return updated_dict

def remove_keys_from_dict(original_dict, keys_to_remove_dict):
    # original_dict에서 keys_to_remove_dict의 키를 제거
    keys_to_remove = set(keys_to_remove_dict.keys())
    # 리스트로 복사하지 않고 직접 순회하며 삭제합니다. 주의: 딕셔너리 크기 변경 중 반복에 주의하세요.
    for key in list(original_dict.keys()):  # list()를 사용하여 딕셔너리 크기 변경 중 오류 방지
        if key in keys_to_remove:
            del original_dict[key]

def copy_files_by_names_dict(file_dict, source_folder, destination_folder):
    # 파일 딕셔너리의 키(파일 이름)를 기반으로 파일을 이동
    for file_name in file_dict.keys():
        # 소스 폴더 내의 파일 경로
        source_file = os.path.join(source_folder, file_name)
        # 목적지 폴더 내의 파일 경로
        destination_file = os.path.join(destination_folder, file_name)
        # 파일을 이동
        shutil.copy(source_file, destination_file)

def copy_files_by_names_list(file_list, source_folder, destination_folder):
    # 파일 리스트 값 (파일 이름)을 기반으로 파일을 이동
    for file_name in file_list:
        # 소스 폴더 내의 파일 경로
        source_file = os.path.join(source_folder, file_name)
        # 목적지 폴더 내의 파일 경로
        destination_file = os.path.join(destination_folder, file_name)
        # 파일을 이동
        shutil.copy(source_file, destination_file)

def copy_origin_to_new_dataset(train_attr_dict, test_attr_dict, query_attr_dict):
    # 별도의 처리가 필요 없는 이미지는 그대로 새로운 데이터셋 폴더에 추가하기
    print(f"Copy ordinary pictures among train images, the number of file: {len(train_attr_dict)}")
    copy_files_by_names_dict(train_attr_dict, train_data_path, new_train_data_path)
    print(f"Copy ordinary pictures among test images, the number of file: {len(test_attr_dict)}")
    copy_files_by_names_dict(test_attr_dict, test_data_path, new_test_data_path)
    print(f"Copy ordinary pictures among query images, the number of file: {len(query_attr_dict)}")
    copy_files_by_names_dict(query_attr_dict, query_data_path, new_query_data_path)

def remove_common_keys(new_label_dict, *other_dicts):
    # 다른 딕셔너리들에서 모든 키를 하나의 세트로 합침
    keys_to_remove = set()
    for d in other_dicts:
        keys_to_remove.update(d.keys())
    
    # new_label_dict에서 keys_to_remove에 해당하는 키를 제거
    for key in list(new_label_dict.keys()):  # list를 사용하여 반복 중 딕셔너리 수정
        if key in keys_to_remove:
            del new_label_dict[key]
    
def filtering_from_origin_dict(new_label_dict, train_attr_dict, test_attr_dict, query_attr_dict):
    # 0번째 값이 0이 아닌 항목만 필터링하여 새로운 딕셔너리 생성
    filtered_dict = {key: value for key, value in new_label_dict.items() if value[0] != 0 and value[0] > 1501}

    # 새로운 인물이 발견되어 번호를 추가해야하는 것들
    train_new_img_dict = check_new_img(filtered_dict, train_attr_dict)
    test_new_img_dict = check_new_img(filtered_dict, test_attr_dict)
    query_new_img_dict = check_new_img(filtered_dict, query_attr_dict)
    # print(train_new_img_dict)


    filtered_dict = {key: value for key, value in new_label_dict.items() if 0 < value[0] and value[0] < 1502}

    # write_dict_to_csv(root_path+"/filtered_dict.csv", filtered_dict)

    # 식별 번호를 다른 번호로 바꾸어야 하는 사진들
    train_change_number_dict = check_new_img(filtered_dict, train_attr_dict)
    test_change_number_dict = check_new_img(filtered_dict, test_attr_dict)
    query_change_number_dict = check_new_img(filtered_dict, query_attr_dict)

    # write_dict_to_csv(root_path + "/test_change_fil_dict.csv", test_change_number_dict)

    train_new_img_dict.update(train_change_number_dict)
    test_new_img_dict.update(test_change_number_dict)
    query_new_img_dict.update(query_change_number_dict)

    filtering_to_remove_dict = {key: value for key, value in new_label_dict.items() if value[0] == -1}

    remove_common_keys(new_label_dict, train_new_img_dict, test_new_img_dict, query_new_img_dict, filtering_to_remove_dict)

    for key in train_new_img_dict.keys():
        if key in new_label_dict.keys():
            print("filtering_from_origin_dict 새로운 라벨 데이터에 잘못된 데이터가 섞임")
            exit()
    # write_dict_to_csv(root_path+"/filtered_new_dict.csv", new_label_dict)

    return new_label_dict, train_new_img_dict, test_new_img_dict, query_new_img_dict

def remove_keys_from_data_dict(train_attr_dict, test_attr_dict, query_attr_dict, train_change_img_dict, test_change_img_dict, query_change_img_dict):
    # 원본 dict에서 특별히 처리해야하는 이미지들은 지우기
    remove_keys_from_dict(train_attr_dict, train_change_img_dict)
    # test
    remove_keys_from_dict(test_attr_dict, test_change_img_dict)
    # query
    remove_keys_from_dict(query_attr_dict, query_change_img_dict)

def change_origin_name_to_new_name(dictionary):
    new_dict = {}
    for key, value in dictionary.items():
        # value[0]을 문자열로 변환하고, 길이가 4가 되도록 앞에 0을 채움
        padded_value0 = str(value[0]).zfill(4)
        # 새로운 키를 생성: value[0]의 4자리 패딩된 값과 key의 나머지 부분 결합
        new_key = padded_value0 + key[4:]
        
        # 새로운 value 리스트 생성: 원래 키를 첫 번째 요소로 설정하고 나머지는 유지
        new_value = [key] + value[1:]  
        
        # 새로운 딕셔너리에 키와 값을 설정
        new_dict[new_key] = new_value
    return new_dict

def copy_files_n_change_name(file_dict, source_folder, destination_folder):
    # 파일 딕셔너리의 키(파일 이름)를 기반으로 파일을 이동
    # 여기선 key가 새로운 파일의 이름, value[0]이 원래 파일의 이름이 된다.
    for new_file_name, value in file_dict.items():
        # 소스 폴더 내의 파일 경로
        source_file = os.path.join(source_folder, value[0])
        # 목적지 폴더 내의 파일 경로
        destination_file = os.path.join(destination_folder, new_file_name)
        # 파일을 이동
        shutil.copy(source_file, destination_file)

def copy_n_change_name(train_change_img_dict, test_change_img_dict, query_change_img_dict):
    '''
    원래 value[0]의 번호로 기존의 사진 이름을 변경해야 한다.
    1. 우선 새로운 이미지 이름을 key로, value[0]에는 원래 이미지 번호를 저장한다.
    2. value[0]의 이름을 가진 파일을 key의 이름으로 복사한다.
    '''

    # 1번 과정 수행
    train_changed_img_dict = change_origin_name_to_new_name(train_change_img_dict)

    test_changed_img_dict = change_origin_name_to_new_name(test_change_img_dict)

    query_changed_img_dict = change_origin_name_to_new_name(query_change_img_dict)

    # 2번 과정 수행
    print(f"Copy ordinary pictures among train images and change name, the number of file: {len(train_changed_img_dict)}")
    copy_files_n_change_name(train_changed_img_dict, train_data_path, new_train_data_path)

    print(f"Copy ordinary pictures among test images and change name, the number of file: {len(test_changed_img_dict)}")
    copy_files_n_change_name(test_changed_img_dict, test_data_path, new_test_data_path)

    print(f"Copy ordinary pictures among query images and change name, the number of file: {len(query_changed_img_dict)}")
    copy_files_n_change_name(query_changed_img_dict, query_data_path, new_query_data_path)

    return train_changed_img_dict, test_changed_img_dict, query_changed_img_dict 

def separate_and_adjust_keys(new_label_dict):
    '''
    jpg가 붙어있는 키와 그렇지 않은 키를 구분하여 리턴
    '''
    with_jpg = {}
    without_jpg = {}
    
    for key, value in new_label_dict.items():
        # '_'가 정확히 3번 등장하는지 확인
        if key.count('_') == 3:
            # '.jpg' 확장자가 있는지 확인
            if '.jpg' not in key:
                key += '.jpg'  # 확장자 추가
        
        # 최종 키를 적절한 딕셔너리에 분류
        if '.jpg' in key:
            with_jpg[key] = value
        else:
            without_jpg[key] = value
            
    return with_jpg, without_jpg

def merge_keys_by_pid(data_dict):
    merged_dict = {}
    not_merged_dict = {}
    # 각 키에 대해 반복
    for key, value in data_dict.items():
        # 키에서 첫 4자리를 새 키로 추출
        new_key = key[:4]
        # 새 키가 병합 딕셔너리에 없다면 추가
        if new_key not in merged_dict:
            merged_dict[new_key] = value
        # 있다면 서로 값을 비교해서 같은 값인지 체크.
        elif merged_dict[new_key] != value:
            not_merged_dict[key] = value
            print(f"without jpg 중 속성이 다른 key: {key}")
    return merged_dict, not_merged_dict

def add_label_normaly(label_dict, file_name_list: list, addition_dict_list: list):
    '''
    train 라벨을 만드는 중이라면
    label_dict
    file_name_dict.keys() = train img names
    key = 추가하려는 라벨의 이름, value = 라벨 값
    '''
    for dict in addition_dict_list:
        for key, value in dict.items():
            # 실제 train 이미지에 매치되는지?
            if key not in file_name_list:
                continue
            label_dict[key] = value[1:]

def add_labels_non_jpg(label_dicts: list, img_name_lists: list, non_jpg_dict: dict):
    matched_names = []
    for i in range(3):
        label_dict = label_dicts[i]
        img_name_list = img_name_lists[i]
        print(i)
        for key, value in non_jpg_dict.items():
            matched_names = []
            # key = 파일 이름인데, 파일 이름이 완벽하지 않아서 매치되는 파일을 찾아야함.
            pattern = key
            print(f"pid: {key}")
            # img_name_list에서 pattern과 일치하는 모든 이름을 찾기
            for img_name in img_name_list:
                if re.match(pattern, img_name):
                    matched_names.append(img_name)
            # print(f"file names: {matched_names}")
            for img_name in matched_names:
                label_dict[img_name] = value[1:]


def create_new_dataset(new_label_dict, train_attr_dict, test_attr_dict, query_attr_dict):

    # 새로운 인물이 발견되어 번호를 추가해야하는 것들 + 식별 번호를 다른 번호로 바꾸어야 하는 사진들
    # 특별히 처리해야 하는 사진 목록을 얻음
    new_label_dict, train_change_img_dict, test_change_img_dict, query_change_img_dict = filtering_from_origin_dict(new_label_dict, train_attr_dict, test_attr_dict, query_attr_dict) 

    if not check_keys_for_jpg(train_change_img_dict):
        print("train에 jpg가 안 붙은게 ..")
        exit()
    if not check_keys_for_jpg(test_change_img_dict):
        print("test에 jpg가 안 붙은게 ..")
        exit()
    if not check_keys_for_jpg(query_change_img_dict):
        print("query에 jpg가 안 붙은게 ..")
        exit()

    # 특별히 처리하지 않아도 되는 것들을 선별
    # remove_keys_from_data_dict(train_attr_dict, test_attr_dict, query_attr_dict, train_change_img_dict, test_change_img_dict, query_change_img_dict)
    # 특별히 처리하지 않아도 되는 것들 이동 
    copy_origin_to_new_dataset(train_attr_dict, test_attr_dict, query_attr_dict)
    # 이름 변경 후 파일 이동 
    # 반환된 dict의 key는 변경된 사진의 이름임
    train_changed_img_dict, test_changed_img_dict, query_changed_img_dict  = copy_n_change_name(train_change_img_dict, test_change_img_dict, query_change_img_dict)

    write_dict_to_csv(root_path+"/train_changed_img_dict.csv", train_changed_img_dict)
    write_dict_to_csv(root_path+"/test_changed_img_dict.csv", test_changed_img_dict)
    write_dict_to_csv(root_path+"/query_changed_img_dict.csv", query_changed_img_dict)

    # 수정된 라벨 만들기
    return create_new_label_csv(new_label_dict, train_changed_img_dict, test_changed_img_dict, query_changed_img_dict )

def create_new_label_csv(new_label_dict, train_change_img_dict, test_change_img_dict, query_change_img_dict):
    train_label_dict = {}
    test_label_dict = {}
    query_label_dict = {}
    # new train 폴더에서 jpg 파일 이름 목록 가져오기
    train_img_name_list = sorted(os.listdir(new_train_data_path))
    train_img_name_list = [file for file in train_img_name_list if file.endswith('.jpg')] 
    # new test 폴더에서 jpg 파일 이름 목록 가져오기
    test_img_name_list = sorted(os.listdir(new_test_data_path))
    test_img_name_list = [file for file in test_img_name_list if file.endswith('.jpg')]
    test_img_name_list = [filename for filename in test_img_name_list if not (filename.startswith('0000') or filename.startswith('-1'))]

    print(test_img_name_list)
    # new query 폴더에서 jpg 파일 이름 목록 가져오기
    query_img_name_list = sorted(os.listdir(new_query_data_path))
    query_img_name_list = [file for file in query_img_name_list if file.endswith('.jpg')] 

    with_jpg, without_jpg = separate_and_adjust_keys(new_label_dict)
    without_jpg, unmerged_dict = merge_keys_by_pid(without_jpg)

    write_dict_to_csv(root_path+"/with_jpg.csv", with_jpg)
    write_dict_to_csv(root_path+"/without_jpg.csv", without_jpg)
    
    # print(train_change_img_dict['0992_c6s2_121693_01.jpg'])
    # print(train_change_img_dict['0991_c6s2_121693_01.jpg'])
    # print(train_change_img_dict)
    # print("\n\n\n\n", test_change_img_dict)

    add_label_normaly(train_label_dict, train_img_name_list, [train_change_img_dict, with_jpg])
    add_label_normaly(test_label_dict, test_img_name_list, [test_change_img_dict, with_jpg])
    add_label_normaly(query_label_dict, query_img_name_list, [query_change_img_dict, with_jpg])

    add_labels_non_jpg([train_label_dict, test_label_dict, query_label_dict], [train_img_name_list, test_img_name_list, query_img_name_list], without_jpg)

    write_dict_to_csv(root_path+"/train_label_dict.csv", train_label_dict)
    write_dict_to_csv(root_path+"/test_label_dict.csv", test_label_dict)
    write_dict_to_csv(root_path+"/query_label_dict.csv", query_label_dict)


    for file_name in train_img_name_list:
        try:
            train_label_dict[file_name]
        except:
            print(f"train label dict don't have key {file_name}")
            exit()


    return train_label_dict, test_label_dict, query_label_dict    

if __name__ == '__main__':
    train_attr_dict =  create_dict_from_filenames(train_data_path, [0 for i in range(28)])
    test_attr_dict =  create_dict_from_filenames(test_data_path, [0 for i in range(28)])
    query_attr_dict =  create_dict_from_filenames(query_data_path, [0 for i in range(28)])

    # test에 들어가는 -1 또는 0000으로 시작하는 이미지 우선 복사
    keys_to_move = [key for key in test_attr_dict.keys() if key.startswith('-1') or key.startswith('0000')]
    # copy_files_by_names_list(keys_to_move, test_data_path, new_test_data_path)

    # '-1' 또는 '0000'으로 시작하는 키 제거
    train_attr_dict = remove_unlabeled_keys(train_attr_dict)
    test_attr_dict = remove_unlabeled_keys(test_attr_dict)
    query_attr_dict = remove_unlabeled_keys(query_attr_dict)

    df = pd.read_excel(correct_label_path, engine='openpyxl')
    df.fillna(0, inplace=True)

    # 'origin_img_name'을 키로, 나머지 컬럼들의 값을 리스트로 가지는 딕셔너리 생성
    new_label_dict = {row['origin_img_name']: row.drop('origin_img_name').astype(int).tolist() for _, row in df.iterrows()}
    # print(new_label_dict['0991_c6s2_121668_01.jpg'])

    # find_diff_label_n_real(new_label_dict, train_attr_dict, test_attr_dict, query_attr_dict)
    train_label_dict, test_label_dict, query_label_dict = create_new_dataset(new_label_dict, train_attr_dict, test_attr_dict, query_attr_dict)

    # print(new_label_dict['1440_c1s6_007541_02.jpg'])
