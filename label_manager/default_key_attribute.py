import pandas as pd
import mat4py

def generate_attribute_dict(dir_path: str, dataset: str):

    mat_attribute_train = mat4py.loadmat(dir_path)[dataset]["train"]
    mat_attribute_train = pd.DataFrame(mat_attribute_train, index=mat_attribute_train['image_index']).astype(int)

    mat_attribute_test = mat4py.loadmat(dir_path)[dataset]["test"]
    mat_attribute_test = pd.DataFrame(mat_attribute_test, index=mat_attribute_test['image_index']).astype(int)

    mat_attribute = mat_attribute_train.add(mat_attribute_test, fill_value=0)
    mat_attribute = mat_attribute.drop(['image_index'], axis=1)
    key_attribute = list(mat_attribute.keys())
    if 'age' in key_attribute:
        key_attribute.remove('age')

    return key_attribute


key_attribute = generate_attribute_dict('/root/amd/label_manager/dataset/Market-1501-v15.09.15/market_attribute.mat', "market_attribute")

# 기존 연구에서 사용하는 키 ATTRIBUTE의 순서
print(key_attribute)

