from multiprocessing import Value
from os import pipe
from torch.utils.tensorboard.writer import SummaryWriter
from our_method.nn_phi import SynNNPhiMeanHelper, SynNNPhiMinHelper
from our_method.nn_psi import SynNNPsiHelper
import utils.common_utils as cu
import numpy as np
import pickle as pkl
from pathlib import Path
import our_method.data_helper as ourdh
import our_method.nn_theta as ournnth 
import our_method.recourse as ourr
import our_method.test_models as tstm
from our_method.methods import BaselineHelper, BaselineKLHelper

cu.set_cuda_device(0)
cu.set_seed(42)

# %% Hyperparamrs and config section
nn_theta = "LR"
models_defname = "-B=2"
syndata_dir = Path("our_method/data/syn/B=2")

budget = 500
num_badex = 10

our_method = "baselinekl"
mh_args = {
    "pretrn_th_phi_psi": {
        "th": False,
        "phi": False,
        "psi": False
    }
}
ourm_epochs = 100 # Number of epochs for our method
tbdir = (f"tblogs/our_method/{models_defname}")


# %% Synthetic Dataset
# Load the Datasets

print("Loading the dataset with B=2")
with open(syndata_dir / "train_3cls.pkl", "rb") as file:
    train = pkl.load(file)
with open(syndata_dir / "test_3cls.pkl", "rb") as file:
    test = pkl.load(file)
with open(syndata_dir / "val_3cls.pkl", "rb") as file:
    val = pkl.load(file)

A = np.array
X, Z, Beta, Y, Ins, Sib, _, _, _ = train
B_per_i = len(Sib[0])
train_data = ourdh.SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        siblings=A(Sib), 
                                        Z_ids=A(Ins))

X, Z,  Beta, Y, Ins = test
test_data = ourdh.SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                    siblings=None, Z_ids=A(Ins))

X, Z,  Beta, Y, Ins = val
val_data = ourdh.SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                    siblings=None, Z_ids=A(Ins))

sdh = ourdh.SyntheticDataHelper(train_data, test_data, val_data)


# %% Synthetic model
if nn_theta == "LR":
    lr_kwargs = {
        "lr": 1e-2
    }
    nnth_mh = ournnth.LRNNthHepler(in_dim=train_data._Xdim, 
                        n_classes=train_data._num_classes,
                        dh = sdh, 
                        **lr_kwargs)
    print("nn_theta is Logistic Regression")
else:
    raise NotImplementedError()

# fit
print("Fitting nn_theta")
nnth_mh.fit_data(epochs=40)
print(f"Accuracy after fitting nn_theta: {nnth_mh.accuracy()}")
nnth_mh.save_model_defname(suffix=models_defname)

# load
nnth_mh.load_model_defname(suffix=models_defname)
print(f"Accuracy of trained nn_theta: {nnth_mh.accuracy()}")
print(f"Grp Accuracy of the ERM model is ")
cu.dict_print(nnth_mh.grp_accuracy())

# %% Synthetic Recourse
cu.set_seed()
synR = ourr.SynRecourseHelper(nnth_mh, sdh, budget=budget, grad_steps=10, num_badex=num_badex)

# fit
synR.recourse_theta()
print(f"Accuracy on last step of Recourse: {synR._nnth.accuracy()}")
synR.dump_recourse_state_defname(suffix=models_defname)

# load
synR.load_recourse_state_defname(suffix=models_defname)
print(f"Accuracy after loading the recourse model is: {synR._nnth.accuracy()}")

# rfit
# synR.nnth_rfit(epochs=1)
# print(f"Accuracy after finetuning nntheta on Recourse set with weighted ERM is {synR._nnth.accuracy()}")
# print(f"Grp Accuracy of the rfit finetuned model is ")
# cu.dict_print(nnth_mh.grp_accuracy())
# synR._nnth.save_model_defname(suffix=models_defname)


# %% NNPhi
nnpihHelper = SynNNPhiMinHelper(in_dim=sdh._train._Xdim+sdh._train._Betadim, out_dim=sdh._train._Betadim,
                            nn_arch=[10, 6], rechlpr=synR, dh=sdh)
# fit
nnpihHelper.fit_rec_beta(epochs=100)
nnpihHelper.save_model_defname(suffix=models_defname)

# load
nnpihHelper.load_model_defname(suffix=models_defname)
pred_betas, aft_acc, bef_acc = nnpihHelper.recourse_accuracy(sdh._test._X, sdh._test._y, sdh._test._Z, sdh._test._Beta)
print(f"Accuracy Before = {bef_acc}; After = {aft_acc}; pred_betas from phi: {np.sum(pred_betas, axis=0)}")



# %% NNPsi
# nnpsiHelper = SynNNPsiHelper(in_dim=sdh._train._Xdim+sdh._train._Betadim, out_dim=1, nn_arch=[10, 6], 
#                             rechlpr=synR, dh=sdh)

# # fit
# nnpsiHelper.fit_rec_r(epochs=20)
# nnpsiHelper.save_model_defname(suffix=models_defname)

# # load
# nnpsiHelper.load_model_defname(suffix=models_defname)
# rid, rec_beta = nnpsiHelper.r_acc(sdh._test._X, sdh._test._y, sdh._test._Beta)
# print(f"Num recourse = {len(rid)}; pred_beta from psi: {np.sum(rec_beta, axis=0)}")



# # %% Assessing three models
# def assess_thphipsi(sdh, nnth, nnphi, nnpsi, pipeline=True):
#     """Assess the three models together

#     Args:
#         sdh ([type]): [description]
#         nnth ([type]): [description]
#         nnphi ([type]): [description]
#         nnpsi ([type]): [description]
#         pipeline (bool, optional): [description]. Defaults to True. This says if we should include recourse only for recourse needed examples.
#     """
#     if pipeline:
#         raw_acc, rec_acc, rs, pred_betas = tstm.assess_th_phi_psi(dh = sdh, nnth=nnth, nnphi=nnphi, 
#                                                                     nnpsi=nnpsi)
#         print(f"Raw acc={raw_acc}, Rec acc={rec_acc}")
#         print(f"Asked recourse for {np.sum(rs)} and predicted betas stats are {np.sum(pred_betas > 0.5, axis=0)}")
#     else:
#         raw_acc, rec_acc, pred_betas = tstm.assess_th_phi(dh = sdh, nnth=nnth, nnphi=nnphi)
#         print(f"Raw acc={raw_acc}, Rec acc={rec_acc}")
#         print(f"predicted betas stats are {np.sum(pred_betas > 0.5, axis=0)}")


# # %% Our method Recourse
# print(f"Initial Accuracy is: {nnth_mh.accuracy()}")
# sw = SummaryWriter(log_dir=f"{tbdir}/{our_method}")
# mh_args = cu.insert_kwargs(mh_args, {"summarywriter": sw})

# if our_method == "baseline":
#     mh = BaselineHelper(dh=sdh, nnth=nnth_mh, nnphi=nnpihHelper, nnpsi=nnpsiHelper, rechlpr=synR, **mh_args)
# elif our_method == "baselinekl":
#     mh = BaselineKLHelper(dh=sdh, nnth=nnth_mh, nnphi=nnpihHelper, nnpsi=nnpsiHelper, rechlpr=synR, **mh_args)
# else:
#     raise ValueError("Please pass a valid our method.")
# print(f"After making the weights to default, accuracy = {nnth_mh.accuracy()}")

# # fit and test
# for epoch in range(ourm_epochs):
#     mh.fit_epoch(epoch=epoch)

#     acc = mh._nnth.accuracy()
#     mh._sw.add_scalar("Epoch_acc", acc, epoch)

#     print(f"Epoch {epoch} Accuracy {acc}")
#     print(f"Epoch {epoch} Grp Accuracy")
#     cu.dict_print(mh.grp_accuracy())
#     raw_acc, rec_acc, rs, pred_betas = tstm.assess_th_phi_psi(dh = sdh, nnth=nnth_mh, nnphi=nnpihHelper, 
#                                                             nnpsi=nnpsiHelper)

#     print("Recourse Accuracy: ")
#     assess_thphipsi(mh._dh, mh._nnth, mh._nnphi, mh._nnpsi, pipeline=False)
