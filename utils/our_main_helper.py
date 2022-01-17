import pickle as pkl
import our_method.methods as ourm
import our_method.constants as constants
import our_method.data_helper as ourdh
import numpy as np
from our_method.methods import MethodsHelper
from our_method.nn_phi import SynNNPhiMinHelper
from our_method.nn_psi import SynNNPsiHelper
import our_method.nn_theta as ournnth
import utils.common_utils as cu
import our_method.recourse as ourr
import our_method.test_models as tstm
import torch
import our_method.data_helper as ourdh

def get_data_helper(dataset_name):
    if dataset_name == constants.SYNTHETIC:
        data_dir = constants.SYN_DIR
        print(f"Loading the dataset: {data_dir}")
        with open(data_dir / "train_3cls.pkl", "rb") as file:
            train = pkl.load(file)
        with open(data_dir / "test_3cls.pkl", "rb") as file:
            test = pkl.load(file)
        with open(data_dir / "val_3cls.pkl", "rb") as file:
            val = pkl.load(file)

        fmt_data_train = lambda ds : (ds[0], ds[1], ds[2], ds[3], ds[4], ds[5], np.ones_like(ds[3])*-1)
        fmt_data_test = lambda ds : (ds[0], ds[1], ds[2], ds[3], None, None, np.ones_like(ds[3])*-1)
        train, test, val = fmt_data_train(train), fmt_data_test(test), fmt_data_test(val) 

    elif dataset_name == constants.SHAPENET_SAMPLE:
        data_dir = constants.SHAPENET_DIR
        with open(data_dir / "data_sample.pkl", "rb") as file:
            shapenet_sample = pkl.load(file)
        data_tuple = []
        for idx in range(7):
            X = np.array([shapenet_sample[entry][idx] for entry in range(900)])
            if idx == 3: # This is to make labels 0-indexed
                X = X-1
            X = np.squeeze(X)
            data_tuple.append(X)
        train, test, val = data_tuple, data_tuple, data_tuple
    elif dataset_name == constants.SHAPENET:
        data_dir = constants.SHAPENET_DIR_SAI
        def process_data(fname):
            with open(data_dir / fname, "rb") as file:
                shapenet_full = pkl.load(file)
            data_tuple = []
            for idx in range(7):
                X = np.array([shapenet_full[entry][idx] for entry in range(len(shapenet_full))])
                X = np.squeeze(X)
                data_tuple.append(X)
            return data_tuple
        train, test, val = process_data("training_shapenet_data.pkl"), process_data("validation_shapenet_data.pkl"), process_data("validation_shapenet_data.pkl")

    A = np.array
    
    if dataset_name == constants.SYNTHETIC:
        X, Z, Beta, Y, Ins, Sib, ideal_betas = train
        B_per_i = len(Sib[0])
        train_data = ourdh.SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                            Siblings=A(Sib), 
                                            Z_ids=A(Ins),
                                            ideal_betas=A(ideal_betas))

        X, Z,  Beta, Y, Ins, _, _ = test
        test_data = ourdh.SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas))

        X, Z,  Beta, Y, Ins, _, _ = val
        val_data = ourdh.SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas))

        dh = ourdh.SyntheticDataHelper(train_data, test_data, val_data)
    
    elif dataset_name == constants.SHAPENET or dataset_name == constants.SHAPENET_SAMPLE:
        X, Z, Beta, Y, Ins, Sib, ideal_betas = train
        B_per_i = len(Sib[0])
        train_args = {
            constants.TRANSFORM: constants.RESNET_TRANSFORMS["train"]
        }
        train_data = ourdh.ShapenetData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                            Siblings=A(Sib), 
                                            Z_ids=A(Ins),
                                            ideal_betas=A(ideal_betas), **train_args)

        test_args = {
            constants.TRANSFORM: constants.RESNET_TRANSFORMS["test"]
        }
        X, Z,  Beta, Y, Ins, _, ideal_betas = test
        test_data = ourdh.ShapenetData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas), **test_args)

        val_args = {
            constants.TRANSFORM: constants.RESNET_TRANSFORMS["val"]
        }
        X, Z,  Beta, Y, Ins, _, ideal_betas = val
        val_data = ourdh.ShapenetData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas), **val_args)
        dh = ourdh.ShapenetDataHelper(train_data, test_data, val_data)
    else:
        raise ValueError("Pass supported datasets only")
    return dh

    

def fit_theta(nn_theta_type, models_defname, dh:ourdh.DataHelper, fit, nnth_epochs, *args, **kwargs):
    if nn_theta_type == constants.LOGREG:
        lr_kwargs = {
            constants.LRN_RATTE: 1e-2
        }
        nnth_mh = ournnth.LRNNthHepler(in_dim=dh._train._Xdim, 
                        n_classes=dh._train._num_classes,
                        dh = dh, 
                        **lr_kwargs)
        print("nn_theta is Logistic Regression")

    elif nn_theta_type == constants.RESNET:
        resnet_kwargs = {
            constants.LRN_RATTE: 1e-3,
            constants.MOMENTUM: 0.9
        }
        resnet_kwargs = cu.insert_kwargs(resnet_kwargs, kwargs)
        nnth_mh = ournnth.ResNETNNthHepler(n_classes=dh._train._num_classes, dh=dh,
                                            **resnet_kwargs)
        print("nn_theta is Resnet Model")
    else:
        raise NotImplementedError()

    # fit
    if fit == True:
        print("Fitting nn_theta")
        nnth_mh.fit_data(epochs=nnth_epochs)
        print(f"Accuracy after fitting nn_theta: {nnth_mh.accuracy()}")
        nnth_mh.save_model_defname(suffix=models_defname)

    # load
    else:
        nnth_mh.load_model_defname(suffix=models_defname)

    # print(f"Accuracy of {nn_theta_type} trained nn_theta: {nnth_mh.accuracy()}")
    # print(f"Grp Accuracy of {nn_theta_type} the ERM model is ")
    # cu.dict_print(nnth_mh.grp_accuracy())
    return nnth_mh


def fit_R_theta(synR:ourr.RecourseHelper, models_defname, epochs=1):
    # rfit
    synR.nnth_rfit(epochs=epochs)
    print(f"Accuracy after finetuning nntheta on Recourse set with weighted ERM is {synR._nnth.accuracy()}")
    print(f"Grp Accuracy of the rfit finetuned model is ")
    cu.dict_print(synR._nnth.grp_accuracy())
    synR._nnth.save_model_defname(suffix=models_defname)



def greedy_recourse(dataset_name, nnth_mh:ournnth.NNthHelper, dh:ourdh.DataHelper, budget, grad_steps, num_badex, models_defname, 
                        fit, *args, **kwargs):
    
    cu.set_seed()

    if dataset_name == constants.SYNTHETIC:
        rechlpr = ourr.SynRecourseHelper(nnth_mh, dh, budget=budget, grad_steps=grad_steps, num_badex=num_badex, *args, **kwargs)
    elif dataset_name == constants.SHAPENET or dataset_name == constants.SHAPENET_SAMPLE:
        rechlpr = ourr.ShapenetRecourseHelper(nnth=nnth_mh, dh = dh, budget=budget, 
                                                grad_steps=grad_steps, num_badex=num_badex, *args, **kwargs)
    # fit
    if fit == True:
        print("Fitting Recourse")
        rechlpr.recourse_theta()
        print(f"Accuracy on last step of Recourse: {rechlpr._nnth.accuracy()}")
        rechlpr.dump_recourse_state_defname(suffix=models_defname)

    # load
    else:
        rechlpr.load_recourse_state_defname(suffix=models_defname)
    
    print(f"Accuracy after loading the recourse model is: {rechlpr._nnth.accuracy()}")
    
    return rechlpr


def fit_nnphi(dh:ourdh.DataHelper, synR:ourr.RecourseHelper, models_defname, fit):
    nnpihHelper = SynNNPhiMinHelper(in_dim=dh._train._Xdim+dh._train._Betadim, out_dim=dh._train._Betadim,
                            nn_arch=[10, 6], rechlpr=synR, dh=dh)

    # fit
    if fit == True:
        print("fitting NPhi")
        nnpihHelper.fit_rec_beta(epochs=100)
        nnpihHelper.save_model_defname(suffix=models_defname)
    # load
    else:
        nnpihHelper.load_model_defname(suffix=models_defname)
    
    pred_betas, aft_acc, bef_acc = nnpihHelper.recourse_accuracy(dh._test._X, dh._test._y, dh._test._Z, dh._test._Beta)
    
    print(f"Accuracy Before = {bef_acc}; After = {aft_acc}; pred_betas from phi: {np.sum(pred_betas, axis=0)}")
    return nnpihHelper


def fit_nnpsi(dh:ourdh.DataHelper, nnarch_args, synR:ourr.RecourseHelper, epochs, models_defname, fit):


    nn_arch = nnarch_args["nn_arch"]
    nnpsiHelper = SynNNPsiHelper(in_dim=dh._train._Xdim+dh._train._Betadim, out_dim=1, nn_arch=nn_arch, 
                                rechlpr=synR, dh=dh)

    # fit
    if fit == True:
        print("Fitting NNPsi")
        nnpsiHelper.fit_rec_r(epochs=epochs)
        nnpsiHelper.save_model_defname(suffix=models_defname)

    # load
    else:
        nnpsiHelper.load_model_defname(suffix=models_defname)

    rid, rec_beta = nnpsiHelper.r_acc(dh._test._X, dh._test._y, dh._test._Beta)
    print(f"Num recourse = {len(rid)}; pred_beta from psi: {np.sum(rec_beta, axis=0)}")
    return nnpsiHelper


# # %% Assessing three models
def assess_thphipsi(dh, nnth, nnphi, nnpsi, pipeline=True):
    """Assess the three models together

    Args:
        sdh ([type]): [description]
        nnth ([type]): [description]
        nnphi ([type]): [description]
        nnpsi ([type]): [description]
        pipeline (bool, optional): [description]. Defaults to True. This says if we should include recourse only for recourse needed examples.
    """
    if pipeline:
        raw_acc, rec_acc, rs, pred_betas = tstm.assess_th_phi_psi(dh = dh, nnth=nnth, nnphi=nnphi, 
                                                                    nnpsi=nnpsi)
        print(f"Raw acc={raw_acc}, Rec acc={rec_acc}")
        print(f"Asked recourse for {np.sum(rs)} and predicted betas stats are {np.sum(pred_betas > 0.5, axis=0)}")
    else:
        raw_acc, rec_acc, pred_betas = tstm.assess_th_phi(dh = dh, nnth=nnth, nnphi=nnphi)
        print(f"Raw acc={raw_acc}, Rec acc={rec_acc}")
        print(f"predicted betas stats are {np.sum(pred_betas > 0.5, axis=0)}")


def get_ourm_hlpr(our_method, dh, nnth, nnphi, nnpsi, synR, **kwargs) -> MethodsHelper:

    if our_method == constants.SEQUENTIAL:
        mh = ourm.BaselineHelper(dh=dh, nnth=nnth, nnphi=nnphi, nnpsi=nnpsi, rechlpr=synR, **kwargs)
    elif our_method == constants.SEQUENTIAL_KL:
        mh = ourm.BaselineKLHelper(dh=dh, nnth=nnth, nnphi=nnphi, nnpsi=nnpsi, rechlpr=synR,  **kwargs)
    elif our_method == constants.METHOD1:
        mh = ourm.Method1Helper(dh=dh, nnth=nnth, nnphi=nnphi, nnpsi=nnpsi, rechlpr=synR, **kwargs)
    else:
        raise ValueError("Please pass a valid our method.")
    print(f"After making the weights to default, accuracy = {nnth.accuracy()}")
    return mh
