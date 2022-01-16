from os import pipe

from torch.utils.tensorboard.writer import SummaryWriter

import our_method.constants as constants
import utils.common_utils as cu
import utils.our_main_helper as main_helper

cu.set_cuda_device(0)
cu.set_seed(42)

freezed_suffixes = []

if __name__ == "__main__":

# %% Hyperparamrs and config section
    nn_theta_type = constants.RESNET
    models_defname = "-resnet"

    assert models_defname not in freezed_suffixes, "Please dont corrupt the freezed models and logs"

    dataset_name = constants.SHAPENET # shapenet_sample, shapenet

    budget = 500
    num_badex = 10
    grad_steps = 10

    tune_theta_R = False

    our_method = constants.METHOD1
    ourm_hlpr_args = {
        constants.PRETRN_THPSIPSI: {
            constants.THETA: False,
            constants.PHI: False,
            constants.PSI: False
        }
    }
    ourm_epochs = 100 # Number of epochs for our method
    tbdir = (constants.TB_DIR / f"our_method/{models_defname}")


# %% Create all the needed objects

    dh = main_helper.get_data_helper(dataset_name = dataset_name)
    
    sw = SummaryWriter(f"tblogs/nn_theta/{models_defname}")
    nnth_args = {
        # constants.SW: sw
    }
    nnth_mh = main_helper.fit_theta(nn_theta_type=nn_theta_type, models_defname=models_defname,
                                            dh = dh, nnth_epochs=50,
                                            fit=False, **nnth_args)
    

    greedy_r = main_helper.greedy_recourse(dataset_name=dataset_name, nnth_mh=nnth_mh, dh=dh, budget=budget, 
                                            grad_steps=grad_steps, num_badex=-1, models_defname=models_defname,
                                            fit = True)

#     if tune_theta_R == True:
#         main_helper.fit_R_theta(synR=greedy_r, models_defname=models_defname)

#     nnphi = main_helper.fit_nnphi(dh=dh, synR=greedy_r, models_defname=models_defname, 
#                                     fit=False)
    
#     psi_arch_args = {
#         "nn_arch": [10, 6]
#     }
#     nnpsi = main_helper.fit_nnpsi(dh=dh, nnarch_args=psi_arch_args, synR=greedy_r, 
#                                     epochs=20, models_defname=models_defname, 
#                                     fit=False)


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
    