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
import numpy as np
from projects.InterpretationReID.interpretationreid.data.datasets.bases import ImageDataset
from fastreid.data.datasets import DATASET_REGISTRY

__all__ = ['Market1501_Interpretation']


# transfer pedestrian attribute to representation in market_attribute.mat
attr2rep = {'long hair': 'hair',
            'T-shirt': 'up',
            'coat': 'up',
            'short of lower-body clothing': 'down',
            'type of lower-body clothing (pants)': 'clothes',
            'wearing hat': 'hat',
            'carrying backpack': 'backpack',
            'carrying bag': 'bag',
            'carrying handbag': 'handbag'}

# the list of empty pedestrian attributes in market_attribute.mat
AttrEmpty = ['wearing boots', 'long coat']

# the list of ambiguous pedestrian attributes in market_attribute.mat
AttrAmbig = ['light color of shoes', 'opening an umbrella', 'pulling luggage',
             'upbrown', 'uppink', 'uporange', 'up mixed colors',
             'downred', 'downorange', 'down mixed colors']

@DATASET_REGISTRY.register()
class Market1501_Interpretation(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    # dataset_dir = 'Market-1587-v24.05.19'
    dataset_dir = ''
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'
    dataset_name = "market1501"

    def __init__(self, root='datasets', market1501_500k=False, **kwargs):
        self.logger = logging.getLogger('fastreid.' + __name__ + 'CORRECT_LABEL')
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = '/root/amd/reid_model/datasets'
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # allow alternative directory structure
        self.data_dir = self.dataset_dir
        # data_dir = osp.join(self.data_dir, 'Market-1587-v24.05.19')
        data_dir = osp.join(self.data_dir, 'Market-1501-v24.05.21_junk_false')
        # data_dir = osp.join(self.data_dir, 'Market-1501-v24.05.21_junk_false')
        if osp.isdir(data_dir):
            self.data_dir = data_dir
        else:
            warnings.warn('The current data structure is deprecated. Please '
                          'put data folders such as "bounding_box_train" under '
                          '"Market1501-1501-v15.09.15".')

        self.train_dir = osp.join(self.data_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.data_dir, 'query')
        self.gallery_dir = osp.join(self.data_dir, 'bounding_box_test')
        self.extra_gallery_dir = osp.join(self.data_dir, 'images')
        self.market1501_500k = market1501_500k

        # self.market_attribute_path = osp.join(self.data_dir, 'market_attribute.mat')
        self.market_train_attr_path = osp.join(self.data_dir, 'train_attribute.csv')
        self.market_test_attr_path = osp.join(self.data_dir, 'test_attribute.csv')
        self.market_query_attr_path = osp.join(self.data_dir, 'query_attribute.csv')
        self.attribute_dict_all = self.generate_attribute_dict(self.market_train_attr_path, self.market_test_attr_path, self.market_query_attr_path,"market_attribute")

        required_files = [
            self.data_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
            self.market_train_attr_path,
            self.market_test_attr_path,
            self.market_query_attr_path
        ]
        if self.market1501_500k:
            required_files.append(self.extra_gallery_dir)
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir)
        query = self.process_dir(self.query_dir, is_train=False)
        gallery = self.process_dir(self.gallery_dir, is_train=False)
        if self.market1501_500k:
            gallery += self.process_dir(self.extra_gallery_dir, is_train=False)

        super(Market1501_Interpretation, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, is_train=True):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))

        pattern = re.compile(r'([-\d]+)_c(\d)')

        data = []
        for img_path in img_paths:
            file_name = osp.basename(img_path)
            pid, camid = map(int, pattern.search(file_name).groups())
            if pid == -1:
                continue  # junk images are just ignored
            assert 0 <= pid <= 1587  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            #print(str(pid))
            if pid == 0:
                p_attribute = -1*torch.ones(size=(26,))
            else:
                p_attribute = self.attribute_dict_all[file_name]

                #p_attribute = p_attribute//p_attribute.abs()
                p_attribute = p_attribute.float()
            if is_train:
                pid = self.dataset_name + "_" + str(pid)
            data.append((img_path, pid, camid,p_attribute))

        return data


    def generate_attribute_dict(self, train_dir_path: str, test_dir_path: str, query_dir_path: str, dataset: str):
        # CSV 파일을 읽어서 데이터프레임으로 변환
        attribute_train = pd.read_csv(train_dir_path)
        attribute_train.set_index('img_name', inplace=True)
    
        attribute_test = pd.read_csv(test_dir_path)
        attribute_test.set_index('img_name', inplace=True)

        attribute_query = pd.read_csv(query_dir_path)
        attribute_query.set_index('img_name', inplace=True)
    
        # train과 test 데이터를 합침
        attribute_df = attribute_train.add(attribute_test, fill_value=0)
        attribute_df = attribute_df.add(attribute_query, fill_value=0)
        attribute_df = attribute_df.astype(int)
        if 'age' in attribute_df.columns:
            attribute_df.drop('age', axis=1, inplace=True)

        # key_attribute 리스트를 갱신
        self.key_attribute = list(attribute_df.columns)

        # dict_attribute를 생성
        dict_attribute = {}
        for idx, row in attribute_df.iterrows():
            if np.isnan(row.values).any() or np.isinf(row.values).any() or (row.values == 0).any():
                print(f"Warning: NaN or infinite value detected in row {idx}, skipping this row.")
                exit()
            dict_attribute[idx] = torch.tensor(row.values.astype(int)) * 2 - 3

        return dict_attribute

    def name_of_attribute(self):
        if self.key_attribute:
            print(self.key_attribute)
            return self.key_attribute
        else:
            assert False