import utils.common_utils as cu
import numpy as np
import pickle as pkl
from pathlib import Path
import our_method.data_helper as ourdh
import our_method.nn_theta as ournnth 

cu.set_cuda_device(0)
cu.set_seed(42)

# %% Hyperparamrs and config section
nn_theta = "LR"


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


if nn_theta == "LR":
    lr_kwargs = {
        "lr": 1e-2
    }
    nnth_mh = ournnth.LRHelper(in_dim=train_data._Xdim, 
                        n_classes=train_data._num_classes,
                        dh = sdh, 
                        **lr_kwargs)
    print("nn_thets is Logistic Regression")
else:
    raise NotImplementedError()

print("Fitting nn_theta")
nnth_mh.fit_data(epochs=40)
print(f"Accuracy after fitting nn_theta: {nnth_mh.accuracy()}")
