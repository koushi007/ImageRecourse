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
from os import replace
from pathlib import Path

import numpy as np
import torch
import utils.common_utils as cu
import utils.torch_utils as tu
from torch._C import dtype
from torch.optim.sgd import SGD
from torch.utils import data

from our_method.data_helper import DataHelper
from our_method.nn_theta import NNthHelper
import our_method.constants as constants


class RecourseHelper(ABC):
    def __init__(self, nnth:NNthHelper, dh:DataHelper, budget, grad_steps=10, num_badex=100, *args, **kwargs) -> None:
        super().__init__()
        self.nnth = nnth
        self.dh = dh
        self.budget = budget
        self.grad_steps = grad_steps
        self.num_badex = num_badex

        self.sgd_optim = SGD([
            {"params": self._nnth._model.parameters()},
        ], lr=1e-3, momentum=0, nesterov=False)
        self.batch_size = 64
        self.lr = 1e-3

        self.R = []
        self.Sij = None
        self.trn_wts = np.ones(self.dh._train._num_data)

        self.__init_kwargs(kwargs)
        self.set_Sij()

    def __init_kwargs(self, kwargs):
        if constants.BATCH_SIZE in kwargs:
            self.batch_size = kwargs[constants.BATCH_SIZE]
        if constants.LRN_RATTE in kwargs:
            self.lr = kwargs[constants.LRN_RATTE]

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
        return Path("our_method/results/syn/models")
    

# %% some utility functions

    def minze_theta(self, rbg_loader, ex_trnwts):
        """Minimizes Theta on the specified data loader
        This does a weighted ERM

        Args:
            rbg_loader ([type]): [description]
            ex_trnwts ([type]): [description]
        """
        self._nnth._model.train()
        for sgd_epoch in range(self._grad_steps):
            for data_ids, zids, x, y, z, b in rbg_loader:
                data_ids, x, y = data_ids.to(cu.get_device(), dtype=torch.int64), x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64)
                self._SGD_optim.zero_grad()

                pred_probs = self._nnth._model.forward_proba(x)
                loss = self._nnth._xecri_perex(pred_probs, y)
                loss = torch.dot(loss, ex_trnwts[data_ids]/loss.size(0))

                loss.backward()
                self._SGD_optim.step()

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
        losses = self._nnth.get_loss_perex(self._dh._train._X, self._dh._train._y)
        if len(R) > 0:
            losses[np.array(R)] = -np.inf
        return heapq.nlargest(num_ex, range(len(losses)), losses.take)

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

    def set_Sij(self):
        """Finds S_ij for every example in TD
        Note, this method does not discriminate if some ids are present in R
        """
        self.Sij = []
        losses = self._nnth.get_loss_perex(self._dh._train._X, self._dh._train._y)
        for data_id in range(self._dh._train._num_data):
            sib_ids = self._dh._train._Siblings[data_id]
            self.Sij.append(np.array(
                (losses[sib_ids] < losses[data_id]) * 1
            ))
        return self.Sij

    def assess_R_candidates(self, trn_wts, R, bad_exs, rbg_loader):
        
        bad_losses = []

        for bad_ex in bad_exs:
                # save the model state
            self._nnth.copy_model()

            # Here we only do SGD Optimizer. Do not use momentum and stuff
            ex_trnwts = self.simulate_addr(trn_wts=trn_wts, R=R, rid=bad_ex)

            self.minze_theta(rbg_loader, torch.Tensor(ex_trnwts).to(cu.get_device()))
            bad_losses.append(self.get_nnth_losses(rbg_loader=rbg_loader))

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
            new_wts[sib_ids] += self.Sij[rid]/sum(self.Sij[rid])
            new_wts[rid] -= 1
            

        # For elements in recourse weights should always be 0 no matter what!
        if len(R) > 0:
            new_wts[np.array(R)] = 0
        return new_wts

    def simulate_rmvr(self, trn_wts, R, rid):
        new_wts = deepcopy(trn_wts)
        sib_ids = self._dh._train._Siblings[rid]
        if sum(self.Sij[rid]) == 0.:
            # If there are no sij, recourse is hopeless
            warnings.warn("Why will Sij be empty when we attempt to add a bad example in R?")
            pass
        else:
            new_wts[rid] += 1
            new_wts[sib_ids] -= self.Sij[rid]/sum(self.Sij[rid])
        if len(R) > 0:
            new_wts[np.array(R)] = 0
        return new_wts

    def get_nnth_losses(self, rbg_loader):
        batch_losses = lambda batchx, batchy: self._nnth.get_loss_perex(batchx, batchy)
        bdexlosses = np.hstack([batch_losses(x, y) for dids, zids, x, y, _, _ in rbg_loader])
        return np.mean(bdexlosses)


    def dump_recourse_state_defname(self, suffix=""):
        dir = self._def_dir
        torch.save(self._nnth._model.state_dict(), dir / f"{self._def_name}{suffix}-nnth.pt")
        with open(dir/f"{self._def_name}{suffix}-R.pkl", "wb") as file:
            pkl.dump(self.R, file)

    def load_recourse_state_defname(self, suffix=""):
        dir = self._def_dir
        self._nnth._model.load_state_dict(
            torch.load(dir/f"{self._def_name}{suffix}-nnth.pt", map_location=cu.get_device())
        )
        with open(dir/f"{self._def_name}{suffix}-R.pkl", "rb") as file:
            self.R = pkl.load(file)
        print(f"Loaded Recourse from {dir}/{self._def_name}{suffix}")

        # Update the Sij based on the new model
        self.set_Sij()
        # Derive the trained weights in accoredance with R and set them
        self.init_trn_wts()

# %% Abstract methods delegated to my children
    @abstractmethod
    def recourse_theta(self, *args, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def nnth_rfit(self, epochs, *args, **kwargs):
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
            self.set_Sij()

            rec_loss = self._nnth.get_loss(bdX, bdy)

            print(f"Inside R iteration {r_iter}; init Loss = {init_loss}; R Loss = {rec_loss}")

        return np.array(self.R), self._Sij

    def nnth_rfit(self, epochs, *args, **kwargs):
        # TODO: maybe we can move this also to ther parent?
        self._nnth.fit_data(loader = self._nnth._trn_loader, 
                            trn_wts=self.trn_wts,
                            epochs=epochs)

    