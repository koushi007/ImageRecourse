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

cu.set_cuda_device(0)
cu.set_seed(42)

# %% Hyperparamrs and config section
nn_theta = "LR"


# %% Synthetic Dataset
# Load the Datasets
syndata_dir = Path("our_method/data/syn")
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

# print("Fitting nn_theta")
# nnth_mh.fit_data(epochs=40)
# print(f"Accuracy after fitting nn_theta: {nnth_mh.accuracy()}")
# nnth_mh.save_model_defname()

nnth_mh.load_model_defname()
print(f"Accuracy of trained nn_theta: {nnth_mh.accuracy()}")
print(f"Grp Accuracy of the ERM model is ")
cu.dict_print(nnth_mh.grp_accuracy())

# %% Synthetic Recourse
cu.set_seed()
synR = ourr.SynRecourseHelper(nnth_mh, sdh, budget=2000, grad_steps=10, num_badex=100)
synR.recourse_theta()
print(f"Accuracy on last step of Recourse: {synR._nnth.accuracy()}")
synR.dump_recourse_state_defname(suffix="-budget=2000")

synR.load_recourse_state_defname()
print(f"Accuracy after loading the recourse model is: {synR._nnth.accuracy()}")
# synR.nnth_rfit(epochs=1)
# print(f"Accuracy after finetuning nntheta on Recourse set with weighted ERM is {synR._nnth.accuracy()}")
# print(f"Grp Accuracy of the rfit finetuned model is ")
# cu.dict_print(nnth_mh.grp_accuracy())
# synR._nnth.save_model_defname(suffix="-rfit")


# %% NNPhi
nnphHelper = SynNNPhiMinHelper(in_dim=sdh._train._Xdim+sdh._train._Betadim, out_dim=sdh._train._Betadim,
                            nn_arch=[10, 6], rechlpr=synR, dh=sdh)
# nnphHelper.fit_rec_beta(epochs=10)
# nnphHelper.save_model_defname()

nnphHelper.load_model_defname()
pred_betas, aft_acc, bef_acc = nnphHelper.recourse_accuracy(sdh._test._X, sdh._test._y, sdh._test._Z, sdh._test._Beta)
print(f"Accuracy Before = {bef_acc}; After = {aft_acc}; pred_betas: {np.sum(pred_betas, axis=0)}")



# %% NNPsi
nnpsiHelper = SynNNPsiHelper(in_dim=sdh._train._Xdim+sdh._train._Betadim, out_dim=1, nn_arch=[10, 6], 
                            rechlpr=synR, dh=sdh)
# nnpsiHelper.fit_rec_r(epochs=20)
# nnpsiHelper.save_model_defname()

nnpsiHelper.load_model_defname()
rid, rec_beta = nnpsiHelper.r_acc(sdh._test._X, sdh._test._y, sdh._test._Beta)
print(f"Num recourse = {len(rid)}; pred_beta: {np.sum(rec_beta, axis=0)}")



# %% Assessing three models
raw_acc, rec_acc, rs, pred_betas = tstm.assess_th_phi_psi(dh = sdh, nnth=nnth_mh, nnph=nnphHelper, 
                                                            nnps=nnpsiHelper)

print(raw_acc, rec_acc)

print(f"Asked recourse for {np.sum(rs)} and predicted betas stats are {np.sum(pred_betas > 0.5, axis=0)}")