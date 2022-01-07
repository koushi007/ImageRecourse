from random import seed
import numpy as np
import torch as torch
from torch import nn
from torch.utils.tensorboard.writer import SummaryWriter
import utils.common_utils as cu
from data_helper import SyntheticDataHelper
from model_helper import LRHelper, Method1Helper, ModelHelper, NNHelper, BaselineHelper
import sys


# Set seed
cu.set_seed(42)
cu.set_cuda_device(0)

# Run configs
pretrn_cls = True # should i load the pretrained classifier?
fit_kwargs = {"interleave_iters": 10} # ho many iterations of interleaved cls/recourse model is needed?
rec_method = "baseline"
suffix = rec_method
num_epochs = 20
print_stats_epoch = True

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
# for epoch in range(10):
#     nnh.fit_epoch(epoch)
#     kwargs = {
#         "Beta": test._Beta
#     }
#     acc = nnh.accuracy(test._X, test._0INDy, **kwargs)
#     nnh._sw.add_scalar("Epoch_Acc", acc, epoch)
#     print(f"Test Accuracy = {acc}")
# nnh.save_model_defname(suffix="-final")
# nnh.load_model_defname(suffix="-final")

# cu.set_seed(42)

# print(f"train accuracy = {nnh.accuracy(train._X, train._0INDy, Beta=train._Beta)}")
# print(f"Test Accuracy = {nnh.accuracy(test._X, test._0INDy, Beta=test._Beta)}")
# print(f"Raw test grp accuracy:")
# cu.dict_print(nnh.grp_accuracy(test._X, test._Beta, test._0INDy))
# sys.exit()


# %% Recourse Model
sw = SummaryWriter(log_dir=f"tblogs/{str(sdh)}/{rec_method}")
kwargs = {
    "summarywriter": sw,
    "batch_size": 50,
}
if rec_method == "baseline":
    rh = BaselineHelper(trn_data=train, tst_data=test, dh=sdh, **kwargs)
elif rec_method == "method1":
    rh = Method1Helper(trn_data=train, tst_data=test, dh=sdh, **kwargs)
else:
    raise NotImplementedError(f"Recourse method: {rec_method} is not supported")

if pretrn_cls:
    rh.load_def_classifier(suffix="-final")
    print(f"Sanity check pretrained classifier: {rh.accuracy()}")

print("Fititng the Recourse Classifier")
for epoch in range(num_epochs):
    rh.fit_epoch(epoch)
    
    acc = rh.accuracy(test._X, test._0INDy, Beta=test._Beta)
    rh._sw.add_scalar("Epoch_Acc", acc, epoch)

    if print_stats_epoch:
        rec_betas = rh.predict_betas(test._X, test._Beta)
        print(np.sum(rec_betas > 0.5, axis=0))

        print(f"Test Accuracy = {acc}")
        print(f"Raw test grp accuracy:")
        cu.dict_print(rh.grp_accuracy(test._X, test._Beta, test._0INDy))

        print(f"Raw train group accuracy is:")
        cu.dict_print(rh.grp_accuracy(train._X, train._Beta, train._0INDy))

        print(f"Recourse Accuracy is: {rh.recourse_accuracy()}")

rh.save_model_defname(suffix=suffix)
rh.load_model_defname(suffix=suffix)

cu.set_seed(42)
print(f"train accuracy = {rh.accuracy(train._X, train._0INDy, Beta=train._Beta)}")
print(f"Test Accuracy = {rh.accuracy(test._X, test._0INDy, Beta=test._Beta)}")
print(f"Raw test grp accuracy:")
cu.dict_print(rh.grp_accuracy(test._X, test._Beta, test._0INDy))
print(f"Raw train group accuracy is:")
cu.dict_print(rh.grp_accuracy(train._X, train._Beta, train._0INDy))
rec_betas = rh.predict_betas(test._X, test._Beta)
print("Sum of all betas asked: ", np.sum(rec_betas > 0.5, axis=0))
print(f"Prob Mean : {np.mean(rec_betas, axis=0)}")

sys.exit()

# %%
