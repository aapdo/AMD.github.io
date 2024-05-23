#!/usr/bin/env python
# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging
import os
import sys


sys.path.append('.')
os.chdir("/root/amd/reid_model") #/home/workspace/로 이동하는것 방지 

from fastreid.config import get_cfg
from projects.InterpretationReID.interpretationreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from projects.InterpretationReID.interpretationreid.evaluation import ReidEvaluator_General, ReidEvaluator
import projects.InterpretationReID.interpretationreid as PII
from fastreid.modeling.meta_arch import build_model
import torch
from fastreid.utils.logger import setup_logger

class Trainer(DefaultTrainer):
    def __init__(cls, cfg, dataset_path):
        cls.dataset_path = dataset_path
        super.__init__(cfg)

    @classmethod
    def build_model(cls, cfg, dataset_path):
        """
        Returns:
            torch.nn.Module:
        It now calls :func:`fastreid.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        cls.dataset_path = dataset_path
        model = build_model(cfg)
        return model    
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`fastreid.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.

        """

        '''
        원래 dataset_name 을 이용해서 add_build_reid_test_loader -> Market1501_Interpretation을 콜하는 방식인데
        dataset_name 대신 dataset_path 를 이용해서 여기서 로드하는 방식으로 바꾸기.
        '''

        return PII.add_build_reid_test_loader_general(cfg, cls.dataset_path)
        
    @classmethod
    def build_evaluator(cls, cfg, num_query, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        # 현재 원래 코드와 동일한 상태.
        # output vector와 attribute 평가를 따로 저장하는 함수.
        # 저장해놓은 vector와 새로운 쿼리랑 비교해서 결과 뽑아주는 함수 만들어야함.
        
        return ReidEvaluator(cfg, num_query)
        # return ReidEvaluator_General(cfg, num_query)
    
    @classmethod
    def test(cls, cfg, model, evaluators=None):
        img_dataset = regist_dataset(cls.dataset_path)
        data_loader, num_query , name_of_attribute = cls.build_test_loader(cfg, img_dataset)
        evaluator = cls.build_evaluator(cfg, num_query=num_query)
        print()

def regist_dataset(dataset_path):
    pid = 0
    camid = 0
    p_attr = torch.full((26,), 0)
    # ( '/root/amd/reid_model/datasezts/Market-1501-v24.05.21_junk_false/bounding_box_test/0330_c5s1_075798_04.jpg', 330, 4, tensor([-1., -1.,  1.,  1.,  1., -1., -1., -1., -1., -1., -1., -1., -1.,  1., 1., -1.,  1.,  1., -1., -1., -1., -1., -1.,  1., -1., -1.]) )
    dataset = []


    return dataset


def setup(args):
    """
    Create configs_old and perform basic setups.
    """
    cfg = get_cfg()
    PII.add_interpretation_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def regist_new_dataset(args, dataset_path):
    args = default_argument_parser().parse_args()
    cfg = setup(args)

    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAIN = False
    model = Trainer.build_model(cfg, dataset_path)

    Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
        
    res = Trainer.test(cfg, model)
    return res

def eval_query():
    print()
    # 모델에 사진 하나 넣고 벡터, 속성 평가 값만 받아오도록 만들고
    # 그걸 리턴해줌. (샐러리 task로)
    # 샐러리 task에서는 데이터베이스 조회해서 갤러리에 있는 애들과 dist 측정하고 
    # 그 중 가장 유사한 10개 뱉어줌. 

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    test_dataset_path = "/root/amd/reid_model/datasets/Market-1501-v24.05.21_junk_false" 
    regist_new_dataset(args, test_dataset_path)
