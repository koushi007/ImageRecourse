from os import pipe

from torch.utils.tensorboard.writer import SummaryWriter

import our_method.constants as constants
import utils.common_utils as cu
import utils.our_main_helper as main_helper
import torch
import numpy as np
import sys

cu.set_cuda_device(3)
cu.set_seed(42)

if __name__ == "__main__":

# %% Hyperparamrs and config section
    nn_theta_type = constants.RESNET
    nntheta_suffix = "-base"
    greedyr_suffix = "-min"
    rec_nnth_suffix = "-rec-min-scratch-sgd1e-2"
    # nnth_fineR_defname = "-resnet-fineR-scratch-sgd"
    nnphi_models_defname = "-resnet-mean"

    dataset_name = constants.SHAPENET # shapenet_sample, shapenet

    budget = 500
    num_badex = 100
    grad_steps = 50
    num_r_per_iter = 10

    tune_theta_R = True # For this we will start with the model that is fit on Shapenet and then finetune it further with the weighted loss function that comes out of recourse
    tune_theta_R_Scratch = True

    our_method = constants.SEQUENTIAL
    ourm_hlpr_args = {
        constants.PRETRN_THPSIPSI: {
            constants.THETA: True,
            constants.PHI: False,
            constants.PSI: False
        }
    }
    ourm_epochs = 100 # Number of epochs for our method
    tbdir = constants.TB_DIR / f"our_method/{nntheta_suffix}"


# %% Create all the needed objects

    dh = main_helper.get_data_helper(dataset_name = dataset_name)
    
    sw = SummaryWriter(constants.TB_DIR / f"nn_theta/{nntheta_suffix}")
    nnth_args = {
        constants.SW: sw,
        constants.BATCH_SIZE: 32
    }
    nnth_mh = main_helper.fit_theta(nn_theta_type=nn_theta_type, models_defname=nntheta_suffix,
                                            dh = dh, nnth_epochs=50,
                                            fit=False, **nnth_args)


    # sys.exit()
    
    sw = SummaryWriter(constants.TB_DIR / f"greedy_rec/{greedyr_suffix}")
    greedy_recargs = {
        constants.SW: sw,
        constants.BATCH_SIZE: 128,
        constants.NUMR_PERITER: num_r_per_iter
    }
    greedy_r = main_helper.greedy_recourse(dataset_name=dataset_name, nnth_mh=nnth_mh, dh=dh, budget=budget, 
                                            grad_steps=grad_steps, num_badex=500, models_defname=greedyr_suffix,
                                            fit = False, **greedy_recargs)

    # sys.exit()

    # sw = SummaryWriter(f"tblogs/fineR_theta/{rec_nnth_suffix}")
    # finetune_nnth_args = {
    #     constants.SW: sw,
    #     constants.SCHEDULER: True,
    #     constants.LRN_RATTE: 1e-2
    # }
    # if tune_theta_R == True:
    #     main_helper.fit_R_theta(synR=greedy_r, scratch=tune_theta_R_Scratch, models_defname=rec_nnth_suffix, epochs=50, **finetune_nnth_args)


    sw = SummaryWriter(f"tblogs/nnphi/{nnphi_models_defname}")
    nnphi_args = {
        constants.SW: sw,
        # constants.SCHEDULER: True
    }
    nnphi = main_helper.fit_nnphi(dataset_name=dataset_name, dh = dh, greedyR=greedy_r, models_defname=nnphi_models_defname, 
                                    fit = True, **nnphi_args)
    pred_betas = nnphi.collect_rec_betas()
    torch.save(pred_betas, "our_method/results/models/nnphi/predbetas-mean-budget=500.pt")  



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
    