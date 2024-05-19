# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import os.path as osp
import re
import warnings
import mat4py
import logging
import pandas as pd
import torch

query_dir_path = '/root/amd/reid_model/datasets/Market-1587-v24.05.18/query_attribute.csv'
test_dir_path = '/root/amd/reid_model/datasets/Market-1587-v24.05.18/test_atrribute.csv'
train_dir_path = '/root/amd/reid_model/datasets/Market-1587-v24.05.18/train_attribute.csv'
dir_path = '/root/amd/reid_model/datasets/Market-1587-v24.05.18/bounding_box_train'

if __name__ == '__main__':
    attribute_train = pd.read_csv(train_dir_path)
    attribute_test = pd.read_csv(test_dir_path)
    attribute_query = pd.read_csv(query_dir_path)
    attribute_train.set_index('img_name', inplace=True)

    attribute_test = pd.read_csv(test_dir_path)
    attribute_test.set_index('img_name', inplace=True)
    
    # train과 test 데이터를 합침
    attribute_df = attribute_train.add(attribute_test, fill_value=0)
    attribute_df = attribute_df.astype(int)
    if 'age' in attribute_df.columns:
        attribute_df.drop('age', axis=1, inplace=True)

    # key_attribute 리스트를 갱신
    key_attribute = list(attribute_df.columns)

    # dict_attribute를 생성
    dict_attribute = {}
    for idx, row in attribute_df.iterrows():
        dict_attribute[idx] = torch.tensor(row.values.astype(int)) * 2 - 3


    img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
    file_names = [osp.basename(path) for path in img_paths]
    
    pattern = re.compile(r'([-\d]+)_c(\d)')
    file_name = '0335_c6s1_077101_02.jpg'

    match = pattern.search(file_name)
    if match:
        pid, camid = map(int, match.groups())
        print(f'PID: {pid}, CamID: {camid}')
    else:
        print("No match found.")