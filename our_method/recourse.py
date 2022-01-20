"""
This file generates the R, S_{ij} using the our method approach.
R = set of indices that are to be recoursed.
S_{ij} = set of indices that are used to recourse.
"""

import heapq
import pickle as pkl
import warnings
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from pathlib import Path
import time

import numpy as np
from sympy import C
import torch
from our_method.models import ResNET
import utils.common_utils as cu
import utils.torch_utils as tu
from torch.optim.sgd import SGD
from torch.optim.adamw import AdamW
from torch.utils import data

from our_method.data_helper import DataHelper
from our_method.nn_theta import NNthHelper
import our_method.constants as constants
from torch.utils.tensorboard.writer import SummaryWriter
import copy


class RecourseHelper(ABC):
    def __init__(self, nnth:NNthHelper, dh:DataHelper, budget, grad_steps=10, num_badex=100, *args, **kwargs) -> None:
        super().__init__()
        self.nnth = nnth
        self.dh = dh
        self.budget = budget
        self.grad_steps = grad_steps

        if num_badex == -1:
            num_badex = dh._train._num_data
        self.num_badex = num_badex

        self.sgd_optim = SGD([
            {"params": self._nnth._model.parameters()},
        ], lr=1e-3, momentum=0, nesterov=False)
        self.batch_size = 32
        self.lr = 1e-3

        self.R = []
        self.Sij = None
        self.trn_wts = np.ones(self.dh._train._num_data)
        self.all_losses_cache = None
        self.num_r_per_iter = 1

        self.__init_kwargs(kwargs)
        self.set_Sij(margin=0)

    def __init_kwargs(self, kwargs):
        if constants.BATCH_SIZE in kwargs.keys():
            self.batch_size = kwargs[constants.BATCH_SIZE]
        if constants.LRN_RATTE in kwargs.keys():
            self.lr = kwargs[constants.LRN_RATTE]
        if constants.SW in kwargs.keys():
            self.sw = kwargs[constants.SW]
        if constants.NUMR_PERITER in kwargs.keys():
            self.num_r_per_iter = kwargs[constants.NUMR_PERITER]

    def init_trn_wts(self):
        for rid in self.R:
            self.trn_wts = self.simulate_addr(trn_wts=self.trn_wts, R=self.R, rid=rid)

# %% Some properties      
    @property
    def _nnth(self):
        return self.nnth
    @_nnth.setter
    def _nnth(self, value):
        self.nnth = value

    @property
    def _dh(self):
        return self.dh
    @_dh.setter
    def _dh(self, value):
        self.dh = value

    @property
    def _budget(self):
        return self.budget
    @_budget.setter
    def _budget(self, value):
        self.budget = value

    @property
    def _grad_steps(self):
        return self.grad_steps
    @_grad_steps.setter
    def _grad_steps(self, value):
        self.grad_steps = value

    @property
    def _num_badex(self):
        return self.num_badex
    @_num_badex.setter
    def _num_badex(self, value):
        self.num_badex = value

    @property
    def _Sij(self):
        return self.Sij
    @_Sij.setter
    def _Sij(self, value):
        self.Sij = value

    @property
    def _R(self):
        return self.R
    @_R.setter
    def _R(self, value):
        self.R = value

    @property
    def _sw(self) -> SummaryWriter:
        return self.sw
    @_sw.setter
    def _sw(self, value):
        self.sw = value

    @property
    def _trn_wts(self):
        return self.trn_wts
    @_trn_wts.setter
    def _trn_wts(self, value):
        self.trn_wts = value

    @property
    def _SGD_optim(self):
        return self.sgd_optim

    @property
    def _lr(self):
        return self.lr
    @property
    def _batch_size(self):
        return self.batch_size

    @property
    def _def_dir(self):
        return Path("our_method/results/models/greedy_rec")
    

# %% some utility functions

    def get_trnloss_perex(self) -> np.array:
        if self.all_losses_cache is not None:
            return self.all_losses_cache
        loader = self._dh._train.get_loader(shuffle=False, batch_size=128) # Have large batch size here for faster parallelism
        batch_losses = lambda batchx, batchy: self._nnth.get_batchloss_perex(batchx, batchy)
        all_losses = np.hstack([batch_losses(x, y) for dids, zids, x, y, _, _ in loader])
        return all_losses

    def sample_bad_ex(self, R:list, num_ex=None):
        """Returns the examples that have the highest loss.

        Note: This only works with training data in dh. I done know if we will ever need to sample bad guys in val/test data?

        Args:
            R ([type]): [description]
            model ([type], optional): [description]. Defaults to None.
            num_ex ([type], optional): [description]. Defaults to None.
        """ 
        if num_ex is None:
            num_ex = self._num_badex

        # We definitely cannot sample elements in R
        num_ex = min(num_ex, self._dh._train._num_data - len(R))

        losses = copy.deepcopy(self.get_trnloss_perex())
        if len(R) > 0:
            losses[np.array(R)] = -np.inf
        return heapq.nlargest(num_ex, range(len(losses)), losses.take)

    def minze_theta(self, loader, ex_trnwts):
        """Minimizes Theta on the specified data loader
        This does a weighted ERM

        Args:
            rbg_loader ([type]): [description]
            ex_trnwts ([type]): [description]
        """
        self._nnth._model.train()
        for data_ids, zids, x, y, z, b in loader:
            data_ids, x, y = data_ids.to(cu.get_device(), dtype=torch.int64), x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64)
            self._SGD_optim.zero_grad()

            pred_probs = self._nnth._model.forward(x)
            loss = self._nnth._xecri_perex(pred_probs, y)
            loss = torch.dot(loss, ex_trnwts[data_ids]/loss.size(0))

            loss.backward()
            self._SGD_optim.step()
        

    def set_Sij(self, margin, loader=None):
        """This method find the set Sij for every ij present in the dataset.
        The recourse algorithm is heavily dependent on the margin. The margin is instrumental in putting only high likelihiood recourse elements into Sij
        You must pass loader with shuffle = False (or) You will hit an run time error.

        We compute losses using the currentr theta of NN_theta and appropriately determine Sij as thoise examples who have
        loss(it) < loss(ij) - margin

        Args:
            margin ([type]): [description]
            loader ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        self.Sij = []
        losses = self.get_trnloss_perex()
        for data_id in self._dh._train._data_ids:
            sib_ids = self._dh._train._Siblings[data_id]
            # This is for average pooling
            # self.Sij.append(np.array(
            #     ( losses[sib_ids] < (losses[data_id]-margin) )*1)
            # )

            # This is for min pooling
            arr = np.zeros(len(sib_ids))
            arr[np.argmin(losses[sib_ids])] = 1 # argmin may be rid itself and I dont care at this point
            self.Sij.append(arr)
        return self.Sij

    def assess_R_candidates(self, trn_wts, R, bad_exs, rbg_loader):
        
        bad_losses = []

        for bad_ex in bad_exs:
                # save the model state
            self._nnth.copy_model()

            # Here we only do SGD Optimizer. Do not use momentum and stuff
            ex_trnwts = self.simulate_addr(trn_wts=trn_wts, R=R, rid=bad_ex)

            self.minze_theta(rbg_loader, torch.Tensor(ex_trnwts).to(cu.get_device()))
            bad_losses.append(np.mean(self.get_trnloss_perex(loader=rbg_loader)))

            self._nnth.apply_copied_model()
            self._nnth.clear_copied_model()
        
        return bad_losses

    def simulate_addr(self, trn_wts, R, rid):
        new_wts = deepcopy(trn_wts)
        sib_ids = self._dh._train._Siblings[rid]
        if sum(self.Sij[rid]) == 0.:
            # If there are no sij, recourse is hopeless
            warnings.warn("Why will Sij be empty when we attempt to add a bad example in R?")
            pass
        else:
            # This is for average pooling

            # numer = []
            # for idx, sid in enumerate(sib_ids):
            #     if  self.Sij[rid][idx] == 1 and sid not in R:
            #         numer.append(1)
            #     else:
            #         numer.append(0)

            # if sum(numer) != 0:
            #     new_wts[sib_ids] += np.array(numer)/sum(numer)
            #     new_wts[rid] -= 1
            # else:
            #     pass

            # This is for min pooling
            new_wts[sib_ids] += self.Sij[rid]
            new_wts[rid] -= 1 # Note if rid happens to be the least in the group, the net effect is no change so that gain=0

        return new_wts
    
    def dump_recourse_state_defname(self, suffix="", model=False):
        dir = self._def_dir
        dir.mkdir(parents=True, exist_ok=True)
        if model:
            self._nnth.save_model_defname(suffix=f"greedy-{suffix}")
        with open(dir/f"{self._def_name}{suffix}-R-Sij.pkl", "wb") as file:
            pkl.dump({"R": self._R, "Sij": self._Sij, "trn_wts": self._trn_wts}, file)

    def load_recourse_state_defname(self, suffix="", model=False):
        dir = self._def_dir
        if model:
            self._nnth.load_model_defname(suffix=f"greedy-{suffix}")
        with open(dir/f"{self._def_name}{suffix}-R-Sij.pkl", "rb") as file:
            rsij_dict = pkl.load(file)
            self._R, self._Sij = rsij_dict["R"], rsij_dict["Sij"] # check if we need to load the trn wts also?
        print(f"Loaded Recourse from {dir}/{self._def_name}{suffix}")

        # Update the Sij based on the new model
        self.set_Sij(margin=0)
        # Derive the trained weights in accoredance with R and set them
        self.init_trn_wts()

# %% Abstract methods delegated to my children
    @abstractmethod
    def recourse_theta(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def nnth_rfit(self, epochs, scratch, *args, **kwargs):
        """nnth_rfit for recourse fit
        This function should be called after recourse.
        Here we fit the nn_theta model on a weighted loss.
        The weights are as decided by the trn_wts here
        We only finetune on the weighted loss

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
        raise NotImplementedError()

    @abstractproperty
    def _def_name(self):
        raise NotImplementedError()


class SynRecourseHelper(RecourseHelper):
    def __init__(self, nnth: NNthHelper, dh: DataHelper, budget, grad_steps=10, num_badex=100, *args, **kwargs) -> None:
        super().__init__(nnth, dh, budget, grad_steps=grad_steps, num_badex=num_badex, *args, **kwargs)
    
    @property
    def _def_name(self):
        return "recourse"

    # This was our first attempt where we sample bad guys and perform min 
    # The new algo however is just to mimic min

    def sample_R_Bd_Gd(self, R, bad_ids, num_req=None, prop = [1, 1, 2]):
        """Samples a dataset to be used later for  computing the gain in loss.

        Args:
            R ([type]): List of ids thar are already added in recourse.
            bad_ids ([type]): [description]
            num_req ([type]): [description]
            prop: 

        Returns:
            sampled ids, Dataloader
        """
        assert len(bad_ids) >= self._dh._train._B_per_i, "Pls pass atleast the required bad_ids"
        num_bdz = int(len(bad_ids)/self._dh._train._B_per_i)
        Rz = np.array(list(set(
            [self._dh._train._Z_ids[entry] for entry in R]
        )))
        bdz = np.array(list(set(
            [self._dh._train._Z_ids[entry] for entry in bad_ids]
        )))

        if num_req is None:
            batch_req = [min(len(Rz), num_bdz*prop[0]), num_bdz, num_bdz*prop[2]]
        else:
            prop = prop/sum(prop)
            batch_req = [min(len(Rz), int(num_req*prop[0])), min(num_bdz, int(num_req*prop[1])), int(num_req*prop[2])]
        
        Rz_smpls = np.random.choice(Rz, batch_req[0], replace=False)
        if len(Rz_smpls) == 0:
            Rz_smpls = Rz_smpls.astype(int)
        bdz_smpls = np.random.choice(bdz, batch_req[1], replace=False)
        bdz_smpls = np.hstack((Rz_smpls, bdz_smpls))
        gdz_smpls = [entry for entry in range(self._dh._train._num_Z) if entry not in bdz_smpls]
        gdz_smpls = np.random.choice(gdz_smpls, batch_req[2], replace=False)
        rbg_zids = np.hstack((bdz_smpls, gdz_smpls))

        ids_smpls, X, y, Z, B = self._dh._train.get_Zinstances(rbg_zids)
        return ids_smpls, X, y,  tu.init_loader(data_ids=ids_smpls, Z_ids=np.repeat(rbg_zids, self._dh._train._B_per_i), 
                                                X=X, y=y, Z=Z, Beta=B, 
                                                shuffle=True, batch_size=self._batch_size)

    def recourse_theta(self, *args, **kwargs):

        trnd = self._dh._train

        def sanity_asserts():
            for r in self.R:
                Sijz = [trnd._Z_ids[entry] for entry in self._Sij[r]]
                assert len(set(Sijz)) == 1, "All the recourse examples should belong to the same object"
                assert Sijz[0] == trnd._Z_ids[r], "The Z_id of r and sij should b consistent"

        # in the begining, all examples have equal loss weights
        self.trn_wts = np.ones(trnd._num_data)

        for r_iter in range(self._budget):
            

            
            bad_exs = self.sample_bad_ex(self.R)

            rbg_ids, bdX, bdy, rbg_loader = self.sample_R_Bd_Gd(self.R, bad_ids=bad_exs, num_req=None, prop=[1,1,2])
            init_loss = self._nnth.get_loss(bdX, bdy)

            bad_losses = self.assess_R_candidates(self.trn_wts, self.R, bad_exs, rbg_loader)

            sel_r = bad_exs[np.argmin(bad_losses)]

            self.R.append(sel_r)
            self.trn_wts = self.simulate_addr(self.trn_wts, self.R, sel_r)
            
            # self._nnth.copy_model()
            self.minze_theta(rbg_loader, torch.Tensor(self.trn_wts).to(cu.get_device()))
            # We will use this Theta going forward as we have already freesed an element in Recourse
            # self._nnth.clear_copied_model()
            self.set_Sij(margin=0)

            rec_loss = self._nnth.get_loss(bdX, bdy)

            print(f"Inside R iteration {r_iter}; init Loss = {init_loss}; R Loss = {rec_loss}")

        return np.array(self.R), self._Sij

    def nnth_rfit(self, epochs, scratch, *args, **kwargs):
        # TODO: maybe we can move this also to ther parent?
        if scratch == True:
            tu.init_weights(self._nnth._model)

        self._nnth.fit_data(loader = self._nnth._trn_loader, 
                            trn_wts=self.trn_wts,
                            epochs=epochs)


class ShapenetRecourseHelper(RecourseHelper):
    def __init__(self, nnth: NNthHelper, dh: DataHelper, budget, grad_steps=10, num_badex=100, *args, **kwargs) -> None:
        super(ShapenetRecourseHelper, self).__init__(nnth, dh, budget, grad_steps, num_badex, *args, **kwargs)

    @property
    def _def_name(self):
        return "shapenet"

    def compute_gain(self, bad_exs, trn_wts) -> np.array:
        """Computes the gain of examples passed in bad_exs

        Args:
            bad_exs ([type]): [description]
        """
        gain = np.ones(len(bad_exs)) * -np.inf
        losses = self.get_trnloss_perex()
        ref_loss = np.dot(trn_wts, losses)

        for idx, rid in enumerate(bad_exs):
            rid_wts = self.simulate_addr(trn_wts=trn_wts, R=self._R, rid=rid)
            recourse_loss = np.dot(rid_wts, losses)
            gain[idx] = ref_loss - recourse_loss

        return gain
    
    def recourse_theta(self, *args, **kwargs):

        # mimic argmin and get gain

        # select the one with highest gain

        # Adjust the trn wts

        # Perform one epoch with full min (better to have some tolerance)

        self.all_losses_cache = self.get_trnloss_perex()

        for r_iter in range(int(self._budget/self.num_r_per_iter)):

            start = time.time()

            self.set_Sij(margin=0)

            bad_exs = self.sample_bad_ex(self.R)

            gain = self.compute_gain(bad_exs=bad_exs, trn_wts=self._trn_wts)

            # sel_r = bad_exs[np.argmax(gain)]

            if self._sw is not None:
                self._sw.add_scalar("gain", np.max(gain), r_iter)

            sel_r =  heapq.nlargest(self.num_r_per_iter, range(len(gain)), gain.take)
            print(f"Gain = {np.mean(gain)}")

            for sel_ridx in sel_r:
                self.R.append(bad_exs[sel_ridx])
                self._trn_wts = self.simulate_addr(self._trn_wts, self.R, bad_exs[sel_ridx])
            
            self.all_losses_cache = None

            self.minze_theta(self._nnth._trn_loader, torch.Tensor(self._trn_wts).to(cu.get_device()))
            
            self.all_losses_cache = self.get_trnloss_perex()

            rec_loss = np.dot(self.get_trnloss_perex(), self._trn_wts)

            print(f"Inside R iteration {r_iter}; Loss after minimizing on adding {len(sel_r)} indices is {rec_loss}", flush=True)

            if self._sw is not None:
                self._sw.add_scalar("Recourse Loss", rec_loss, r_iter)

            if (r_iter+1) % 100 == 0:
                self.dump_recourse_state_defname(suffix=f"riter-{r_iter}", model=False)

            print(f"Time taken = {time.time() - start}")

        self.all_losses_cache = None

        return np.array(self.R), self._Sij, self._trn_wts
    
    def nnth_rfit(self, scratch, epochs, *args, **kwargs):
        
        rfit_args = {}
        
        if scratch == True:
            # Initialize the model with pretrained resnet weights
            self._nnth._model = ResNET(out_dim=self._nnth._trn_data._num_classes, **self._nnth.kwargs)
            self._nnth._model = self._nnth._model.to(cu.get_device())

            lr = 1e-3
            if constants.LRN_RATTE in kwargs.keys():
                lr = kwargs[constants.LRN_RATTE]

            rfit_args[constants.OPTIMIZER] = SGD([
                                                {'params': self._nnth._model.parameters()},
                                            ], lr=lr)
        
        else:
            rfit_args[constants.OPTIMIZER] = AdamW([
                                                {'params': self._nnth._model.parameters()},
                                            ], lr=1e-5)

        rfit_args = cu.insert_kwargs(rfit_args, kwargs)
        
        self._nnth.fit_data(loader = self._nnth._trn_loader, 
                            trn_wts=self.trn_wts,
                            epochs=epochs, **rfit_args)