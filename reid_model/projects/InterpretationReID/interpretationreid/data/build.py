# encoding: utf-8
"""
@author:  l1aoxingyu
@contact: sherlockliao01@gmail.com
"""

import os
import torch
# instead of import container_abcs
import collections.abc
# instead of import string_classes
string_classes = str
# instead of import int_classes
int_classes = int
from torch.utils.data import DataLoader
from fastreid.utils import comm

#from fastreid.data import samplers
from . import samplers
from .common import CommDataset
from fastreid.data.datasets import DATASET_REGISTRY
from fastreid.data.transforms import build_transforms

_root = os.getenv("FASTREID_DATASETS", "datasets")


def add_build_reid_train_loader(cfg):
    cfg = cfg.clone()
    cfg.defrost()

    train_items = list()
    for d in cfg.DATASETS.NAMES:
        dataset = DATASET_REGISTRY.get(d)(root=_root, combineall=cfg.DATASETS.COMBINEALL)
        if comm.is_main_process():
            dataset.show_train()
        train_items.extend(dataset.train)

    iters_per_epoch = len(train_items) // cfg.SOLVER.IMS_PER_BATCH
    cfg.SOLVER.MAX_ITER *= iters_per_epoch
    train_transforms = build_transforms(cfg, is_train=True)
    train_set = CommDataset(train_items, train_transforms, relabel=True)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    num_instance = cfg.DATALOADER.NUM_INSTANCE
    mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // comm.get_world_size()

    if cfg.DATALOADER.PK_SAMPLER:
        if cfg.DATALOADER.NAIVE_WAY:
            if cfg.DATALOADER.ATT_RESAMPLE:
                data_sampler = samplers.NaiveIdentitySamplerForCrossDomain(train_set.img_items,
                                                         cfg.SOLVER.IMS_PER_BATCH, num_instance, cfg=cfg)
            else:
                data_sampler = samplers.NaiveIdentitySampler(train_set.img_items,
                                                             cfg.SOLVER.IMS_PER_BATCH, num_instance)
        else:
            assert False
            data_sampler = samplers.BalancedIdentitySampler(train_set.img_items,
                                                            cfg.SOLVER.IMS_PER_BATCH, num_instance)
    else:
        assert False
        data_sampler = samplers.TrainingSampler(len(train_set))
    batch_sampler = torch.utils.data.sampler.BatchSampler(data_sampler, mini_batch_size, True)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return train_loader


def add_build_reid_test_loader(cfg, dataset_name):
    cfg = cfg.clone()
    cfg.defrost()

    dataset = DATASET_REGISTRY.get(dataset_name)(root=_root)
    if comm.is_main_process():
        dataset.show_test()
    test_items = dataset.query + dataset.gallery
    test_transforms = build_transforms(cfg, is_train=False)
    test_set = CommDataset(test_items, test_transforms, relabel=False)

    mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=0,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return test_loader, len(dataset.query) , dataset.name_of_attribute()

def add_build_reid_test_loader_general(cfg, test_items):
    cfg = cfg.clone()
    cfg.defrost()
    name_of_attribute = ['backpack', 'bag', 'clothes', 'down', 'downblack', 'downblue', 'downbrown', 'downgray', 'downgreen', 'downpink', 'downpurple', 'downwhite', 'downyellow', 'gender', 'hair', 'handbag', 'hat', 'up', 'upblack', 'upblue', 'upgray', 'upgreen', 'uppurple', 'upred', 'upwhite', 'upyellow']

    test_transforms = build_transforms(cfg, is_train=False)
    test_set = CommDataset(test_items, test_transforms, relabel=False)

    # 수정 필요 없음
    mini_batch_size = cfg.TEST.IMS_PER_BATCH // comm.get_world_size()
    data_sampler = samplers.InferenceSampler(len(test_set))
    batch_sampler = torch.utils.data.BatchSampler(data_sampler, mini_batch_size, False)
    
    test_loader = DataLoader(
        test_set,
        batch_sampler=batch_sampler,
        num_workers=0,  # save some memory
        collate_fn=fast_batch_collator,
        pin_memory=True,
    )
    return test_loader, 0, name_of_attribute

def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def fast_batch_collator(batched_inputs):
    """
    A simple batch collator for most common reid tasks
    """
    elem = batched_inputs[0]
    if isinstance(elem, torch.Tensor):
        out = torch.zeros((len(batched_inputs), *elem.size()), dtype=elem.dtype)
        for i, tensor in enumerate(batched_inputs):
            out[i] += tensor
        return out

    elif isinstance(elem, collections.abc.Mapping):
        return {key: fast_batch_collator([d[key] for d in batched_inputs]) for key in elem}

    elif isinstance(elem, float):
        return torch.tensor(batched_inputs, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batched_inputs)
    elif isinstance(elem, string_classes):
        return batched_inputs
