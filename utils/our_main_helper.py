import pickle as pkl
from baseline.data_helper import DataHelper
import our_method.constants as constants
import our_method.data_helper as ourdh
import numpy as np
import our_method.nn_theta as ournnth
import utils.common_utils as cu

def get_data_helper(dataste_name):
    if dataste_name == constants.SYNTHETIC:
        data_sir = constants.SYN_DIR
        print(f"Loading the dataset: {data_sir}")
        with open(data_sir / "train_3cls.pkl", "rb") as file:
            train = pkl.load(file)
        with open(data_sir / "test_3cls.pkl", "rb") as file:
            test = pkl.load(file)
        with open(data_sir / "val_3cls.pkl", "rb") as file:
            val = pkl.load(file)

        A = np.array
        X, Z, Beta, Y, Ins, Sib, _, _, _ = train
        ideal_betas = np.ones_like(Y) * -1
        B_per_i = len(Sib[0])
        train_data = ourdh.SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                            siblings=A(Sib), 
                                            Z_ids=A(Ins),
                                            ideal_betas=A(ideal_betas))

        X, Z,  Beta, Y, Ins = test
        test_data = ourdh.SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas))

        X, Z,  Beta, Y, Ins = val
        val_data = ourdh.SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas))

        sdh = ourdh.SyntheticDataHelper(train_data, test_data, val_data)
        return sdh

def fit_theta(nn_theta_type, models_defname, dh:DataHelper, fit=True, nnth_epochs=40):
    if nn_theta_type == "LR":
        lr_kwargs = {
        "lr": 1e-2
    }
        nnth_mh = ournnth.LRNNthHepler(in_dim=dh._train._Xdim, 
                        n_classes=dh._train._num_classes,
                        dh = dh, 
                        **lr_kwargs)
        print("nn_theta is Logistic Regression")
    else:
        raise NotImplementedError()

    # fit
    if fit == True:
        print("Fitting nn_theta")
        nnth_mh.fit_data(epochs=40)
        print(f"Accuracy after fitting nn_theta: {nnth_mh.accuracy()}")
        nnth_mh.save_model_defname(suffix=models_defname)

    # load
    else:
        nnth_mh.load_model_defname(suffix=models_defname)

    print(f"Accuracy of {nn_theta_type} trained nn_theta: {nnth_mh.accuracy()}")
    print(f"Grp Accuracy of {nn_theta_type} the ERM model is ")
    cu.dict_print(nnth_mh.grp_accuracy())
    return nnth_mh