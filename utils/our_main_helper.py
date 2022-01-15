import pickle as pkl
from baseline.data_helper import DataHelper
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


def get_data_helper(dataset_name):
    if dataset_name == constants.SYNTHETIC:
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
                                            Siblings=A(Sib), 
                                            Z_ids=A(Ins),
                                            ideal_betas=A(ideal_betas))

        X, Z,  Beta, Y, Ins = test
        test_data = ourdh.SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas))

        X, Z,  Beta, Y, Ins = val
        val_data = ourdh.SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas))

        sdh = ourdh.SyntheticDataHelper(train_data, test_data, val_data)
        return sdh

def fit_theta(nn_theta_type, models_defname, dh:DataHelper, fit, nnth_epochs):
    if nn_theta_type == constants.LOGREG:
        lr_kwargs = {
        constants.LRN_RATTE: 1e-2
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
        nnth_mh.fit_data(epochs=nnth_epochs)
        print(f"Accuracy after fitting nn_theta: {nnth_mh.accuracy()}")
        nnth_mh.save_model_defname(suffix=models_defname)

    # load
    else:
        nnth_mh.load_model_defname(suffix=models_defname)

    print(f"Accuracy of {nn_theta_type} trained nn_theta: {nnth_mh.accuracy()}")
    print(f"Grp Accuracy of {nn_theta_type} the ERM model is ")
    cu.dict_print(nnth_mh.grp_accuracy())
    return nnth_mh


def fit_R_theta(synR:ourr.RecourseHelper, models_defname):
    # rfit
    synR.nnth_rfit(epochs=1)
    print(f"Accuracy after finetuning nntheta on Recourse set with weighted ERM is {synR._nnth.accuracy()}")
    print(f"Grp Accuracy of the rfit finetuned model is ")
    cu.dict_print(synR._nnth.grp_accuracy())
    synR._nnth.save_model_defname(suffix=models_defname)



def greedy_recourse(nnth_mh:ournnth.NNthHelper, dh:DataHelper, budget, grad_steps, num_badex, models_defname, fit):
    cu.set_seed()
    synR = ourr.SynRecourseHelper(nnth_mh, dh, budget=budget, grad_steps=grad_steps, num_badex=num_badex)

    # fit
    if fit == True:
        print("Fitting Recourse")
        synR.recourse_theta()
        print(f"Accuracy on last step of Recourse: {synR._nnth.accuracy()}")
        synR.dump_recourse_state_defname(suffix=models_defname)

    # load
    else:
        synR.load_recourse_state_defname(suffix=models_defname)
    
    print(f"Accuracy after loading the recourse model is: {synR._nnth.accuracy()}")
    
    return synR


def fit_nnphi(dh:DataHelper, synR:ourr.RecourseHelper, models_defname, fit):
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


def fit_nnpsi(dh:DataHelper, nnarch_args, synR:ourr.RecourseHelper, epochs, models_defname, fit):


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
