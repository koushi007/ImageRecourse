import warnings
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from pathlib import Path
from random import shuffle

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import sampler
import utils.common_utils as cu
import utils.torch_utils as tu
from sklearn.metrics import accuracy_score
from torch.optim import SGD, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import our_method.data_helper as ourdh
from our_method.data_helper import Data
from our_method.models import FNNXBeta, LRModel, ResNETPsi
from our_method.nn_theta import NNthHelper
from our_method.recourse import RecourseHelper
import our_method.constants as constants


class NNPsiHelper(ABC):
    def __init__(self, psimodel:nn.Module, rechlpr:RecourseHelper, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        super().__init__()
        self.psimodel = psimodel
        self.rechlpr = rechlpr
        self.dh = dh
        
        self.lr = 1e-3
        self.sw = None
        self.batch_size = 16
        self.lr_scheduler = None

        self.__init_kwargs(kwargs)
        self.__init_loaders()
    
    def __init_kwargs(self, kwargs:dict):
        if constants.LRN_RATTE in kwargs.keys():
            self.lr = kwargs[constants.LRN_RATTE]
        if constants.SW in kwargs.keys():
            self.sw = kwargs[constants.SW]
        if constants.BATCH_SIZE in kwargs.keys():
            self.batch_size = kwargs[constants.BATCH_SIZE]

    def __init_loaders(self):
        # This is very simple. We just need x, y, R in each batch. Thats all
        # here care needs to be taken so that each batch has 50% +ve samples

        R_ids = self._rechlpr._R
        trn_X = self._dh._train._X
        trn_Beta = self._dh._train._Beta
        trn_tgts = np.zeros(len(trn_X), dtype=int)
        trn_tgts[np.array(R_ids)] = 1
        trn_tgts_oh = np.array([1-trn_tgts, trn_tgts]).T
        batch_sampler = tu.MultilabelBalancedRandomSampler(trn_tgts_oh) 

        psi_dsargs = {
            constants.TRANSFORM: self._dh._train.transform
        }
        T = torch.Tensor
        phi_ds = tu.CustomPsiDataset(X=T(trn_X), Beta=T(trn_Beta), R_tgts=T(trn_tgts_oh), **psi_dsargs)
        self.trn_loader = data_utils.DataLoader(phi_ds, batch_size=self._batch_size, sampler=batch_sampler)

        self.tst_loader = self._dh._test.get_loader(shuffle=False, batch_size=self._batch_size)
        self.val_loader = self._dh._val.get_loader(shuffle=False, batch_size=self._batch_size)

# %% Properties
    @property
    def _psimodel(self) -> FNNXBeta:
        return self.psimodel
    @_psimodel.setter
    def _psimodel(self, value):
        self.psimodel = value

    @property
    def _rechlpr(self) -> RecourseHelper:
        return self.rechlpr
    @_rechlpr.setter
    def _rechlpr(self, value):
        self.rechlpr = value

    @property
    def _dh(self):
        return self.dh
    @_dh.setter
    def _dh(self, value):
        self.dh = value

    @property
    def _batch_size(self):
        return self.batch_size

    @property
    def _trn_loader(self):
        return self.trn_loader

    @property
    def _tst_loader(self):
        return self.tst_loader

    @property
    def _val_loader(self):
        return self.val_loader

    @property
    def _trn_data(self) -> Data:
        return self.dh._train
    @_trn_data.setter
    def _trn_data(self, value):
        raise ValueError("Why are u setting the data object once again?")   

    @property
    def _tst_data(self) -> Data:
        return self._dh._test
    @_tst_data.setter
    def _tst_data(self, value):
        raise ValueError("Why are u setting the data object once again?")   

    @property
    def _val_data(self) -> Data:
        return self._dh._val
    @_val_data.setter
    def _val_data(self, value):
        raise ValueError("Why are u setting the data object once again?")

    @property
    def _optimizer(self):
        if  self.optimizer == None:
            raise ValueError("optimizer not yet set")
        return self.optimizer
    @_optimizer.setter
    def _optimizer(self, value):
        self.optimizer = value

    @property
    def _lr(self) -> nn.Module:
        return self.lr
    @_lr.setter
    def _lr(self, value):
        self.lr = value

    @property
    def _sw(self) -> SummaryWriter:
        return self.sw
    @_sw.setter
    def _sw(self, value):
        self.sw = value

    @property
    def _lr_scheduler(self):
        return self.lr_scheduler
    @_lr_scheduler.setter
    def _lr_scheduler(self, value):
        self.lr_scheduler = value

    @property
    def _def_dir(self):
        return Path("our_method/results/models/nnpsi")

    @abstractproperty
    def _def_name(self):
        raise NotImplementedError()

    @property
    def _xecri(self):
        return nn.CrossEntropyLoss()

    @property
    def _xecri_perex(self):
        return nn.CrossEntropyLoss(reduction="none")
    
    @property
    def _bcecri(self):
        return nn.BCELoss()
    
    @property
    def _bcecri_perex(self):
        return nn.BCELoss(reduction="none")

    @property
    def _msecri(self):
        return nn.MSELoss()

    @property
    def _msecri_perex(self):
        return nn.MSELoss(reduction="none")

# %% Abstract methods to be delegated to my children
    @abstractmethod
    def fit_rec_r(self, epochs, loader:data_utils.DataLoader, *args, **kwargs):
        """Fits the Recourse beta that was learned during recourse R, Sij generation

        Args:
            loader (data_utils.DataLoader): [description]
            epochs ([type]): [description]

        Raises:

        Returns:
            [type]: [description]
        """
        raise NotImplementedError()
       
# %% Some utilities
    def r_acc(self, data="train", *args, **kwargs) -> float:
        """Returns the Accuracy of predictions

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]

        Returns:
            predicted rids and Betas that were asked to recourse
        """
        if data != "train":
            raise ValueError("Code what to do for test/val as we dont knoe the r targets")

        loader = data_utils.DataLoader(self._trn_loader.dataset, batch_size=128, shuffle=False)
        corrects = 0  
        
        self._psimodel.eval()

        rpreds_all = []
        with torch.no_grad():
            for epoch_step, (X, Beta, R) in enumerate(loader):
                X, Beta, R = X.to(cu.get_device()), Beta.to(cu.get_device(), dtype=torch.int64), R.to(cu.get_device())
                R = torch.argmax(R, dim=1).squeeze().to(dtype=torch.float32) # We did one hot encoding for sampler needs. We convert this to labels here again

                self._optimizer.zero_grad()
                rpreds = self._psimodel.forward_r(X, Beta).squeeze()

                rpreds = rpreds > 0.5
                
                correct = torch.sum(rpreds == R)
                corrects += correct
        return corrects / len(loader.dataset)

    def predict_r(self, X_test, Beta_test):
        raise NotImplementedError()
        self._psimodel.eval()
        with torch.no_grad():
            X_test, Beta_test = torch.Tensor(X_test).to(cu.get_device()), torch.Tensor(Beta_test).to(cu.get_device())
            return self._psimodel.forward_r(X_test, Beta_test).cpu().numpy()


    def save_model_defname(self, suffix=""):
        dir = self._def_dir
        dir.mkdir(exist_ok=True, parents=True)
        fname = dir / f"{self._def_name}{suffix}.pt"
        torch.save(self._psimodel.state_dict(), fname)
    
    def load_model_defname(self, suffix=""):
        fname = self._def_dir / f"{self._def_name}{suffix}.pt"
        print(f"Loaded model from {str(fname)}")
        self._psimodel.load_state_dict(torch.load(fname, map_location=cu.get_device()))

class SynNNPsiHelper(NNPsiHelper):  
    """This is the default class for BBPhi.
    This des mean Recourse of Betas

    Args:
        NNPhiHelper ([type]): [description]
    """
    def __init__(self, in_dim, out_dim, nn_arch:list, rechlpr:RecourseHelper, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nn_arch = nn_arch

        assert (in_dim, out_dim) == (dh._train._Xdim+dh._train._Betadim, 1), "Why are the input and output dimensions fo NNPhi inconsistent?"

        # if u need dropouts, pass it in kwargs
        psimodel = FNNXBeta(in_dim=in_dim, out_dim=out_dim, nn_arch=nn_arch, prefix="psi")
        super(SynNNPsiHelper, self).__init__(psimodel, rechlpr, dh, args, kwargs)

        tu.init_weights(self._psimodel)
        self._psimodel.to(cu.get_device())
        

        self._optimizer = AdamW([
            {'params': self._psimodel.parameters()},
        ], lr=self._lr)
        
    def fit_rec_r(self, epochs, loader:data_utils.DataLoader=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader): [description]
            epochs ([type], optional): [description]. Defaults to None. 
        """
        global_step = 0
        self._psimodel.train()
        if loader is None:
            loader = self._trn_loader
        else:
            warnings.warn("Are u sure u want to pass a custom loader?")

        for epoch in range(epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (X, Beta, R) in enumerate(loader):
                global_step += 1

                X, Beta, R = X.to(cu.get_device()), Beta.to(cu.get_device(), dtype=torch.int64), R.to(cu.get_device(), dtype=torch.int64)
                R = torch.argmax(R, dim=1) # We did one hot encoding for sampler needs. We convert this to labels here again

                self._optimizer.zero_grad()
                rpreds = self._psimodel.forward(X, Beta)
                loss = self._bcecri(rpreds, R)

                loss.backward()
                self._optimizer.step()
                tq.set_postfix({"Loss": loss.item()})
                tq.update(1)                
        

    @property
    def _def_name(self):
        return f"nnpsi_{self.nn_arch}"


class ShapenetNNPsiHelper(NNPsiHelper):  
    """This is the default class for BBPhi.
    This des mean Recourse of Betas

    Args:
        NNPhiHelper ([type]): [description]
    """
    def __init__(self, out_dim, nn_arch:list, rechlpr:RecourseHelper, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        self.out_dim = out_dim
        self.nn_arch = nn_arch

        assert out_dim == 1, "R network shoud predict onl one value for (x, beta)"

        # if u need dropouts, pass it in kwargs
        psimodel = ResNETPsi(out_dim=out_dim, nn_arch=nn_arch, beta_dims=dh._train._BetaShape, prefix="psi", *args, **kwargs)
        super(ShapenetNNPsiHelper, self).__init__(psimodel, rechlpr, dh, *args, **kwargs)

        self._psimodel.to(cu.get_device())
        
        self._optimizer = AdamW([
            {'params': self._psimodel.parameters()},
        ], lr=self._lr)
        
    def fit_rec_r(self, epochs, loader:data_utils.DataLoader=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader): [description]
            epochs ([type], optional): [description]. Defaults to None. 
        """
        global_step = 0
        self._psimodel.train()
        if loader is None:
            loader = self._trn_loader
        else:
            warnings.warn("Are u sure u want to pass a custom loader?")
        
        if constants.SCHEDULER in kwargs.keys():
            self._lr_scheduler = tu.get_lr_scheduler(self._optimizer, scheduler_name="cosine_annealing", n_rounds=epochs)

        for epoch in range(epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (X, Beta, R) in enumerate(loader):
                global_step += 1

                X, Beta, R = X.to(cu.get_device()), Beta.to(cu.get_device(), dtype=torch.int64), R.to(cu.get_device())
                R = torch.argmax(R, dim=1).squeeze().to(dtype=torch.float32) # We did one hot encoding for sampler needs. We convert this to labels here again

                self._optimizer.zero_grad()
                rpreds = self._psimodel.forward_r(X, Beta).squeeze()
                loss = self._bcecri(rpreds, R)

                loss.backward()
                self._optimizer.step()
                tq.set_postfix({"Loss": loss.item()})

                if self._sw is not None:
                    self._sw.add_scalar("Psi_Loss", loss.item(), global_step)

                tq.update(1)                

            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

    @property
    def _def_name(self):
        return "psi"

