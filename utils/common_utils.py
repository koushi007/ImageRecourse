import random
import torch
import numpy as np
from copy import deepcopy
import json
import torch.nn as nn

def set_seed(seed:int=42):
    """Sets the seed for torch, numpy and random
    Args:
        seed (int): [description]
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_device():
    return "cuda"

def set_cuda_device(gpu_num: int):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

def insert_kwargs(kwargs:dict, new_args:dict):
    assert type(new_args) == type(kwargs), "Please pass two dictionaries"
    merged_args = kwargs.copy()
    merged_args.update(new_args)
    return merged_args

def dict_print(d:dict):
    print(json.dumps(d, sort_keys=False, indent=4))

def init_weights(m:nn.Module):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
