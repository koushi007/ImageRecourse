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
from torch._C import dtype
from torch.optim import SGD, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import our_method.data_helper as ourdh
from our_method.data_helper import Data
from our_method.models import FNN, LRModel
from our_method.nn_theta import NNthHelper
from our_method.recourse import RecourseHelper


class NNPsiHelper(ABC):
    def __init__(self, psimodel:nn.Module, rechlpr:RecourseHelper, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        super().__init__()
        self.psimodel = psimodel
        self.rechlpr = rechlpr
        self.dh = dh
        
        self.lr = 1e-3
        self.sw = None
        self.batch_size = 16

        self.__init_kwargs(kwargs)
        self.__init_loaders()
    
    def __init_kwargs(self, kwargs:dict):
        if "lr" in kwargs.keys():
            self.lr = kwargs["lr"]
        if "summarywriter" in kwargs:
            self.sw = kwargs["summarywriter"]
        if "batch_size" in kwargs:
            self.batch_size = kwargs["batch_size"]

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
        self.trn_loader = tu.generic_init_loader(trn_X, trn_Beta, trn_tgts, batch_size=self._batch_size, sampler=batch_sampler)

        self.tst_loader = self._dh._test.get_loader(shuffle=False, batch_size=self._batch_size)
        self.val_loader = self._dh._val.get_loader(shuffle=False, batch_size=self._batch_size)

# %% Properties
    @property
    def _psimodel(self) -> FNN:
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
    def _def_dir(self):
        return Path("our_method/results/syn/models")

    @abstractproperty
    def _def_name(self):
        raise NotImplementedError()

    @property
    def _lr(self):
        return self.lr

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
    def r_acc(self, X_test, y_test, Beta_test, *args, **kwargs) -> float:
        """Returns the Accuracy of predictions

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]

        Returns:
            predicted rids and Betas that were asked to recourse
        """
        pred_r = self.predict_r(X_test, Beta_test)
        pred_r = (pred_r > 0.5) * 1
        pred_rids = np.where(pred_r == 1)[0]
        pred_recBeta = Beta_test[pred_rids]
        return pred_rids, pred_recBeta

    def predict_r(self, X_test, Beta_test):
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
        psimodel = FNN(in_dim=in_dim, out_dim=out_dim, nn_arch=nn_arch, prefix="psi")
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

                X, Beta, R = X.to(cu.get_device()), Beta.to(cu.get_device()), R.to(cu.get_device())

                self._optimizer.zero_grad()
                rpreds = self._psimodel.forward_r(X, Beta)
                loss = self._bcecri(rpreds, R)

                loss.backward()
                self._optimizer.step()
                tq.set_postfix({"Loss": loss.item()})
                tq.update(1)                
        

    @property
    def _def_name(self):
        return f"nnpsi_{self.nn_arch}"

