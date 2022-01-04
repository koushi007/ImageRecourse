from random import seed
import numpy as np
import torch as torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
import utils.common_utils as cu
from data_helper import SyntheticDataHelper
from model_helper import LRHelper, ModelHelper, NNHelper, RecourseHelper
import sys


# Set seed
cu.set_seed(42)
cu.set_cuda_device(0)


# Dataset HyperParameters
dim=10 
prior=0.5
num_train=1000 
B_per_i=10
num_test=1000 
beta_noise=0.7 
mean_start=0.5 
mean_step=0.1 
var_start=0.05 
var_step=0.1

# %% Sample Dataset
sdh = SyntheticDataHelper(dim=dim, prior=prior, num_train=num_train, B_per_i=B_per_i,
                            num_test=num_test, beta_noise=beta_noise, 
                            mean_start=mean_start, mean_step=mean_step, 
                            var_start=var_start, var_step=var_step)
train, test = sdh._train, sdh._test


# %% LR Model
# lr_cls = LRHelper(trn_data=train, tst_data=test, dh=sdh, max_iter=100)
# lr_cls.fit()
# print(f"Raw Train Accuracy: {lr_cls.accuracy(train._X, train._0INDy)}")
# print(f"Raw test accuracy: {lr_cls.accuracy(test._X, test._0INDy)}")
# print(f"Raw test grp accuracy:")
# cu.pretty_print(lr_cls.grp_accuracy(test._X, test._Beta, test._0INDy))
# print(f"Classifier Weights: {lr_cls._weights}")
# sys.exit()

# %% NN ClS model
# sw = SummaryWriter(log_dir=f"tblogs/{str(sdh)}/nn_cls")
# kwargs = {
#     "summarywriter": sw,
#     "batch_size": 50,
# }
# nnh = NNHelper(trn_data=train, tst_data=test, dh=sdh, **kwargs)
# # for epoch in range(50):
# #     nnh.fit_epoch(epoch)
# #     kwargs = {
# #         "Beta": test._Beta
# #     }
# #     acc = nnh.accuracy(test._X, test._0INDy, **kwargs)
# #     nnh._sw.add_scalar("Epoch_Acc", acc, epoch)
# #     print(f"Test Accuracy = {acc}")
# # nnh.save_model_def()
# nnh.load_model_defname()

# cu.set_seed(42)

# print(f"train accuracy = {nnh.accuracy(train._X, train._0INDy, Beta=train._Beta)}")
# print(f"Test Accuracy = {nnh.accuracy(test._X, test._0INDy, Beta=test._Beta)}")
# print(f"Raw test grp accuracy:")
# cu.pretty_print(nnh.grp_accuracy(test._X, test._Beta, test._0INDy))
# sys.exit()


# %% Recourse Model
sw = SummaryWriter(log_dir="tblogs/recourse")
kwargs = {
    "summarywriter": sw,
    "batch_size": 50,
}
rh = RecourseHelper(trn_data=train, tst_data=test, dh=sdh, **kwargs)
for epoch in range(50):
    rh.fit_epoch(epoch)

    acc = rh.accuracy(test._X, test._0INDy, Beta=test._Beta)
    rh._sw.add_scalar("Epoch_Acc", acc, epoch)
    print(f"Test Accuracy = {acc}")

rh.save_model_def()

rh.load_model_defname()
cu.set_seed(42)

print(f"train accuracy = {rh.accuracy(train._X, train._0INDy, Beta=train._Beta)}")
print(f"Test Accuracy = {rh.accuracy(test._X, test._0INDy, Beta=test._Beta)}")
print(f"Raw test grp accuracy:")
cu.pretty_print(rh.grp_accuracy(test._X, test._Beta, test._0INDy))
sys.exit()

# %%
