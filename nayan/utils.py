import pprint
import logging
import constants
from logging.handlers import QueueHandler
import torch
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from config_new import config
import pandas as pd
from scipy.stats import norm
from logistic_regression import *
import pickle as pkl

import sys
import math


# The below 2 functions are used to find the overlap area of the 2 gaussaian distributions
def solve(m1,m2,std1,std2):
  a = 1/(2*std1**2) - 1/(2*std2**2)
  b = m2/(std2**2) - m1/(std1**2)
  c = m1**2 /(2*std1**2) - m2**2 / (2*std2**2) - np.log(std2/std1)
  return np.roots([a,b,c])


def overlap(m1,std1,m2,std2):

    #Get point of intersect
    result = solve(m1,m2,std1,std2)
    r = result[0]
    # integrate
    area = ( norm.cdf(r,m2,std2)  ) 
    # print("Area under curves ", area)
    return area

# This function will find the Sij for the samples selected into R
# It will also find the weights_list
def find_sij(new_R_list,w1,b1,w2,b2,w3,b3,y,process_train_path):

    # load the processed train data
    with open(process_train_path, "rb") as file:
        l = pkl.load(file)
    x_list, z_list, beta_list, label_list, instance_list, sibling_list, R_list, weights_list, sij_list = l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]

    R_list = new_R_list
    weights_list = 1 - new_R_list

    for i in range(new_R_list.shape[0]):
       
        loss_i = find_loss(torch.Tensor(x_list[i:i+1]),w1,b1,w2,b2,w3,b3,torch.LongTensor(y[i:i+1]))
        loss_siblings = find_loss(torch.Tensor(x_list)[sibling_list[i]],w1,b1,w2,b2,w3,b3,torch.LongTensor(y)[sibling_list[i]])
        sij_list[i] = (loss_siblings<loss_i).to(int)

        if new_R_list[i]==1:

            if torch.sum(sij_list[i]) == 0.0:
                # If no Sij is present
                weights_list[i] = weights_list[i] + 1.0

            else:
                # If Sij is present then distribute the weight equally
                weights_list[sibling_list[i]] = weights_list[sibling_list[i]] + sij_list[i]/(torch.sum(sij_list[i])+1e-8)

    return weights_list, sij_list

# This will select onbly the best Sij
def find_best_sij(new_R_list,w1,b1,w2,b2,w3,b3,y,process_train_path):


     # load the processed train data
    with open(process_train_path, "rb") as file:
        l = pkl.load(file)
    x_list, z_list, beta_list, label_list, instance_list, sibling_list, R_list, weights_list, sij_list = l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]

    R_list = new_R_list
    weights_list = 1 - new_R_list

    for i in range(new_R_list.shape[0]):
       
        loss_i = find_loss(torch.Tensor(x_list[i:i+1]),w1,b1,w2,b2,w3,b3,torch.LongTensor(y[i:i+1]))
        loss_siblings = find_loss(torch.Tensor(x_list)[sibling_list[i]],w1,b1,w2,b2,w3,b3,torch.LongTensor(y)[sibling_list[i]])

        t = torch.tensor([0 for i in range(loss_siblings.shape[0])])
        t[torch.argmin(loss_siblings-loss_i)] = 1

        sij_list[i] = t

        if new_R_list[i]==1:

            if torch.sum(sij_list[i]) == 0.0:
                # If no Sij is present
                weights_list[i] = weights_list[i] + 1.0

            else:
                weights_list[sibling_list[i]] = weights_list[sibling_list[i]] + sij_list[i]/(torch.sum(sij_list[i])+1e-8)

    return weights_list, sij_list


# This fucntion will calculate the loss
def find_loss(X,w1,b1,w2,b2,w3,b3,y):

    first = torch.mm(X, w1) + b1
    second = torch.mm(X, w2) + b2
    third = torch.mm(X, w3) + b3
    
    y_hat = torch.softmax(torch.cat([first,second,third],axis = 1),dim=1)

    loss = torch.nn.CrossEntropyLoss(reduce=False)(y_hat,y)

    return loss


def pretty_print(object):
    """Pretty prints the given object and returns the string
    """
    return pprint.pformat(object, indent=4)

def init_logger(file_name, file_mode="w"):

    # set up logging to file
    logging.basicConfig(filename=file_name, filemode=file_mode,
                    format=constants.LOG_FORMAT, level=logging.DEBUG)

    logger = logging.getLogger(file_name)
    h = logging.handlers.SysLogHandler()
    h.setLevel(logging.ERROR)

    logger.addHandler(h)
    return logger

def make_grid(imgs:torch.Tensor, file_name:str=None, title="title"):
    """Makes a pytorch grid of images and saves it to the file in logs/images directory

    Args:
        imgs ([numpy images]): List of numpy images
    """
    batch_tensor = torch.stack([torch.from_numpy(i) for i in imgs],0)
    # print("shape = ",batch_tensor.shape)
    grid_img = torchvision.utils.make_grid(batch_tensor, nrow=2)
    # print("grid shape = ",grid_img.shape)

    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    if file_name is not None:
        plt.savefig(constants.root_dir/ config["plots_dir"] / file_name)
    plt.show()






def find_bin_disp(lower, upper, data):
    """Finds the bin id and the displacement within the bin
    Args:
        lower ([array]): [description]
        upper ([arras]): [description]
        data ([Float]): [description]

    Returns:
        ID of the bin, displacement within the bin in [0,1]
    """
    bin_ids = []
    disp = []
    for entry in data:
        for idx, (l, u) in enumerate(zip(lower, upper)):
            if entry >= l and entry <= u:
                bin_ids.append(idx)
                disp.append((entry - l)/ (u - l))
                break
    return torch.LongTensor(bin_ids), torch.squeeze(torch.FloatTensor(disp))


def find_inc_dec(preds, targets, tolerance):
    """Finds if the predictions should increase or decrease to match the targets

    Args:
        preds ([type]): [description]
        targets ([type]): [description]
    Returns:
        Binary ==> 
        2 means it should increase
        1 means do nothing
        0 means it should decrease
    """
    inc_dec = []
    for pred_i, tgt_i in zip(preds, targets):
        diff = tgt_i - pred_i
        if diff <= -tolerance:
            inc_dec.append(0)
        elif diff >= tolerance:
            inc_dec.append(2)
        else:
            inc_dec.append(1)
    return torch.LongTensor(inc_dec)


def mid(lower, upper, bin_ids):
    """Gets the midpoint of tyhe bin based on the binids

    Args:
        lower ([type]): [description]
        upper ([type]): [description]
        bin_id ([type]): [description]

    Returns:
        [list]: mid points
    """
    mid = []
    for entry in bin_ids:
        mid.append((lower[entry] + upper[entry])/2)

    return torch.squeeze(torch.FloatTensor(mid))

def get_inc_dec_beta(beta:torch.Tensor, reg_preds:dict, tol=0.05):
    """Transforms the beta according to the predictions
    if pred == 0, then decrease by tol
    if pred == 1, do nothing
    if pred == 2, increase by tol

    Args:
        beta ([type]): [description]
        reg_preds ([type]): [description]
    """
    for entry in constants.delta_beta_order:
        preds = reg_preds[entry]
        _, preds = torch.max(preds, dim=1)
        preds = preds.numpy().tolist()

        [beta[constants.delta_beta_idx[entry]]][preds == 0] -= tol
        [beta[constants.delta_beta_idx[entry]]][preds == 2] += tol

        # beta[:, constants.delta_beta_idx[entry]][preds == 0] -= tol
        # beta[:, constants.delta_beta_idx[entry]][preds == 2] += tol

    return beta


def dump_csv(cols, col_names, file_path="Experiments/table.csv"):
    """Dumps the datafraem with the columns and the correspoiding column names

    Args:
        cols ([type]): list of lists
        col_names ([type]): list of corresponding column names
        file_path (str, optional): [description]. Defaults to "Experiments/table.csv".
    """ 
    num_samples = len(cols[0])
    if len(cols) > 1:
        for idx, entry in enumerate(cols[1:], start=1):
            assert len(entry) == num_samples, f"column {col_names[0]} has #{num_samples} but column {col_names[idx]} had #{len(entry)}"
    df_dict = {}
    for col, col_name in zip(cols, col_names):
        df_dict[col_name] = col
    df = pd.DataFrame(df_dict)
    df.to_csv(file_path, index=True)
    return df


if __name__ == "__main__":

    # Dump csv
    col1 = [1,2,3,4]
    col2 = [5,6,7,8]
    cols = [col1, col2]
    col_names = ["col1", "col2"]
    dump_csv(cols, col_names)