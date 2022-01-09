import random
import torch
import numpy as np
from copy import deepcopy
import json
import torch.nn as nn
import torch.utils.data as data_utils

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

def init_loader(data_ids, Z_ids, X, y, Z, Beta, shuffle=True, batch_size=None):
        T = torch.Tensor
        dataset = data_utils.TensorDataset(T(data_ids), T(Z_ids), T(X), T(y), T(Z), T(Beta))
        return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def init_grp_loader(data_ids, Z_ids, X, y, Z, Beta, B_per_i, shuffle=True, batch_size=None):
    grp_arr = lambda arr : np.array(np.split(arr, int(len(arr) / B_per_i)))
    return init_loader(grp_arr(data_ids), grp_arr(Z_ids), grp_arr(X), grp_arr(y), grp_arr(Z), grp_arr(Beta), 
                            shuffle=shuffle, batch_size=int(batch_size / B_per_i))

def generic_init_loader(*args, **kwargs):
    """This is a generic init loader. We just create a dataset of any which crap u send to us
    Use the above init_loader to ensure some discipline.
    But pls pass batch size and shuffle as keargs in the very least
    """
    shuffle = kwargs["shuffle"]
    bsz = kwargs["batch_size"]
    T = torch.Tensor
    dataset = data_utils.TensorDataset(*[T(entry) for entry in args])
    return data_utils.DataLoader(dataset, shuffle=shuffle, batch_size=bsz)
