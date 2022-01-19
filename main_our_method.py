from os import pipe

from torch.utils.tensorboard.writer import SummaryWriter

import our_method.constants as constants
import utils.common_utils as cu
import utils.our_main_helper as main_helper
import torch
import numpy as np
import sys

cu.set_cuda_device(1)
cu.set_seed(42)

if __name__ == "__main__":

# %% Hyperparamrs and config section
    nn_theta_type = constants.RESNET
    nntheta_models_defname = "-resnet"
    rec_models_defname = "-min"
    # rec_nntheta_defname = "-debug-rectheta-ft5epochs-adam1e-5"
    # nnth_fineR_defname = "-resnet-fineR-scratch-sgd"
    # nnphi_models_defname = "-resnet-budget=500"

    dataset_name = constants.SHAPENET # shapenet_sample, shapenet

    budget = 500
    num_badex = 1000
    grad_steps = 50

    tune_theta_R = False # For this we will start with the model that is fit on Shapenet and then finetune it further with the weighted loss function that comes out of recourse
    tune_theta_R_Scratch = False

    our_method = constants.SEQUENTIAL
    ourm_hlpr_args = {
        constants.PRETRN_THPSIPSI: {
            constants.THETA: True,
            constants.PHI: False,
            constants.PSI: False
        }
    }
    ourm_epochs = 100 # Number of epochs for our method
    tbdir = constants.TB_DIR / f"our_method/{nntheta_models_defname}"


# %% Create all the needed objects

    dh = main_helper.get_data_helper(dataset_name = dataset_name)
    
    sw = SummaryWriter(f"tblogs/nn_theta/{nntheta_models_defname}")
    nnth_args = {
        # constants.SW: sw
    }
    nnth_mh = main_helper.fit_theta(nn_theta_type=nn_theta_type, models_defname=nntheta_models_defname,
                                            dh = dh, nnth_epochs=50,
                                            fit=False, **nnth_args)
    

    greedy_r = main_helper.greedy_recourse(dataset_name=dataset_name, nnth_mh=nnth_mh, dh=dh, budget=budget, 
                                            grad_steps=grad_steps, num_badex=1000, models_defname=rec_models_defname,
                                            fit = True)

    sys.exit()

    # sw = SummaryWriter(f"tblogs/rec_nn_theta/{rec_nntheta_defname}")
    # recnnth_args = {
    #     constants.SW: sw
    # }
    # if tune_theta_R == True:
    #     main_helper.fit_R_theta(synR=greedy_r, scratch=tune_theta_R_Scratch, models_defname=nnth_fineR_defname, epochs=5)


    # sw = SummaryWriter(f"tblogs/nnphi/{rec_nntheta_defname}")
    # nnphi_args = {
    #     constants.SW: sw
    # }
    # nnphi = main_helper.fit_nnphi(dataset_name=dataset_name, dh = dh, greedyR=greedy_r, models_defname=nnphi_models_defname, 
    #                                 fit = True, **nnphi_args)
    # tgt_betas = nnphi.collect_tgt_betas()
    # torch.save(tgt_betas, "our_method/results/models/nnphi/tgtbetas-budget=500.pt")



    # psi_arch_args = {
    #     "nn_arch": [10, 6]
    # }
    # nnpsi = main_helper.fit_nnpsi(dh=dh, nnarch_args=psi_arch_args, synR=greedy_r, 
    #                                 epochs=20, models_defname=nnphi_models_defname, 
    #                                 fit=False)


# # %% Kick starts our method training

#     print(f"Initial Accuracy is: {nnth_mh.accuracy()}")
#     sw = SummaryWriter(log_dir=f"{tbdir}/{our_method}")
#     ourm_hlpr_args = cu.insert_kwargs(ourm_hlpr_args, {constants.SW: sw})

#     ourm_hlpr = main_helper.get_ourm_hlpr(our_method=our_method, dh=dh, nnth=nnth_mh, 
#                                             nnphi=nnphi, nnpsi=nnpsi, synR=greedy_r, **ourm_hlpr_args)

#     # fit and test
#     for epoch in range(ourm_epochs):
#         ourm_hlpr.fit_epoch(epoch=epoch)

#         acc = ourm_hlpr._nnth.accuracy()
#         ourm_hlpr._sw.add_scalar("Epoch_acc", acc, epoch)

#         print(f"Epoch {epoch} Accuracy {acc}")
#         print(f"Epoch {epoch} Grp Accuracy")
#         cu.dict_print(ourm_hlpr.grp_accuracy())
#         main_helper.assess_thphipsi(ourm_hlpr._dh, ourm_hlpr._nnth, ourm_hlpr._nnphi, ourm_hlpr._nnpsi)

#         print("Recourse Accuracy: ")
#         main_helper.assess_thphipsi(ourm_hlpr._dh, ourm_hlpr._nnth, ourm_hlpr._nnphi, ourm_hlpr._nnpsi, pipeline=False)
    