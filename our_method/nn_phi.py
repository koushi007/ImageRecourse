import warnings
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import utils.common_utils as cu
import utils.torch_utils as tu
from sklearn.metrics import accuracy_score
from torch.optim import SGD, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import our_method.data_helper as ourdh
from our_method.data_helper import Data
from our_method.models import FNN, LRModel
from our_method.nn_theta import NNthHelper
from our_method.recourse import RecourseHelper
import our_method.constants as constants


class NNPhiHelper(ABC):
    def __init__(self, phimodel:nn.Module, rechlpr:RecourseHelper, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        super().__init__()
        self.phimodel = phimodel
        self.rechlpr = rechlpr
        self.dh = dh
        
        self.lr = 1e-3
        self.sw = None
        self.batch_size = 16

        self.__init_kwargs(kwargs)
        self.__init_loaders()
    
    def __init_kwargs(self, kwargs:dict):
        if constants.LRN_RATTE in kwargs.keys():
            self.lr = kwargs[constants.LRN_RATTE]
        if constants.SW in kwargs:
            self.sw = kwargs[constants.SW]
        if constants.BATCH_SIZE in kwargs:
            self.batch_size = kwargs[constants.BATCH_SIZE]

    def __init_loaders(self):
        # Initializing train loader is fairly complicated
        # tst_loader and val_loader behave as always

        # For training, we need X, Beta, Sib_beta, Sij, losses of siblings (to implement many strategies)
        # To avoid complications, we only pass Beta ans tgt beta shall be computed runtime based on Sij
        trn_X = self._trn_data._X
        trn_y = self._trn_data._y
        trn_Beta = self._trn_data._Beta
        trn_Sij = np.array(self._rechlpr._Sij)
        trn_sibs = self._trn_data._Siblings
        trn_losses = self._rechlpr._nnth.get_loss_perex(trn_X, trn_y)
        sib_losses = np.array([trn_losses[sib_ids] for sib_ids in trn_sibs])
        # now get the target beta. We only compute the sibnling betas here. Create the target as please later
        trn_sib_beta = np.array([
            trn_Beta[sib_ids] for sib_ids in trn_sibs
        ])
        
        R_ids = np.array(self._rechlpr._R)
        self.trn_loader = tu.generic_init_loader(R_ids, trn_X[R_ids], trn_Beta[R_ids], trn_sib_beta[R_ids], trn_Sij[R_ids], sib_losses[R_ids], 
                shuffle=True, batch_size=self._batch_size)

        self.tst_loader = self._dh._test.get_loader(shuffle=False, batch_size=self._batch_size)
        self.val_loader = self._dh._val.get_loader(shuffle=False, batch_size=self._batch_size)

# %% Properties
    @property
    def _phimodel(self):
        return self.phimodel
    @_phimodel.setter
    def _phimodel(self, value):
        self.phimodel = value

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
    def _xecri(self):
        return nn.CrossEntropyLoss()

    @property
    def _xecri_perex(self):
        return nn.CrossEntropyLoss(reduction="none")
    
    @property
    def _msecri(self):
        return nn.MSELoss()

    @property
    def _msecri_perex(self):
        return nn.MSELoss(reduction="none")

# %% Abstract methods to be delegated to my children
    @abstractmethod
    def fit_rec_beta(self, epochs, loader:data_utils.DataLoader, *args, **kwargs):
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
    def recourse_accuracy(self, X_test, y_test, Z_test, Beta_test, recourse_theta=0.5, *args, **kwargs) -> float:
        """Returns the Accuracy of predictions

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]

        Returns:
            before accuracy, predicted betas, after accuracy
        """
        pred_beta = self.predict_beta(X_test, Beta_test)
        pred_beta = (pred_beta > recourse_theta) * 1
        X_test_rec = np.multiply(Z_test, pred_beta)
        return pred_beta, self._rechlpr._nnth.accuracy(X_test_rec, y_test), self._rechlpr._nnth.accuracy(X_test, y_test)

    def predict_beta(self, X_test, Beta_test):
        self._phimodel.eval()
        with torch.no_grad():
            X_test, Beta_test = torch.Tensor(X_test).to(cu.get_device()), torch.Tensor(Beta_test).to(cu.get_device())
            return self._phimodel.forward(X_test, Beta_test).cpu().numpy()

    def get_loss_perex(self, X_test, y_test):
        """Gets the cross entropy loss per example

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]

        Returns:
            np.array of losses
        """
        T = torch.Tensor
        X_test, y_test = T(X_test), T(y_test).to(cu.get_device(), dtype=torch.int64)
        probs = self.predict_proba(X_test)
        raise NotImplementedError("Decide the correct loss here") # TODO
        return self._xecri_perex(T(probs).to(cu.get_device()), y_test).cpu().numpy()

    def get_loss(self, X_test, y_test):
        """Returns the Loss of the batch

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
        T = torch.Tensor
        X_test, y_test = T(X_test), T(y_test).to(cu.get_device(), dtype=torch.int64)
        probs = self.predict_proba(X_test)
        raise NotImplementedError("Decide the correct loss here") # TODO
        return self._xecri(T(probs).to(cu.get_device()), y_test).cpu().numpy()

    def save_model_defname(self, suffix=""):
        dir = self._def_dir
        dir.mkdir(exist_ok=True, parents=True)
        fname = dir / f"{self._def_name}{suffix}.pt"
        torch.save(self._phimodel.state_dict(), fname)
    
    def load_model_defname(self, suffix=""):
        fname = self._def_dir / f"{self._def_name}{suffix}.pt"
        print(f"Loaded model from {str(fname)}")
        self._phimodel.load_state_dict(torch.load(fname, map_location=cu.get_device()))

class SynNNPhiMeanHelper(NNPhiHelper):  
    """This is the default class for BBPhi.
    This des mean Recourse of Betas

    Args:
        NNPhiHelper ([type]): [description]
    """
    def __init__(self, in_dim, out_dim, nn_arch:list, rechlpr:RecourseHelper, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nn_arch = nn_arch

        assert (in_dim, out_dim) == (dh._train._Xdim+dh._train._Betadim, dh._train._Betadim), "Why are the input and output dimensions fo NNPhi inconsistent?"

        # if u need dropouts, pass it in kwargs
        phimodel = FNN(in_dim=in_dim, out_dim=out_dim, nn_arch=nn_arch, prefix="phi")
        super(SynNNPhiMeanHelper, self).__init__(phimodel, rechlpr, dh, args, kwargs)

        tu.init_weights(self._phimodel)
        self._phimodel.to(cu.get_device())
        

        self._optimizer = AdamW([
            {'params': self._phimodel.parameters()},
        ], lr=self._lr)
        
    def fit_rec_beta(self, epochs, loader:data_utils.DataLoader=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader): [description]
            epochs ([type], optional): [description]. Defaults to None. 
        """
        global_step = 0
        self._phimodel.train()
        if loader is None:
            loader = self._trn_loader
        else:
            warnings.warn("Are u sure u want to pass a custom loader?")

        for epoch in range(epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (rids, X, Beta, SibBeta, Sij, Siblosses) in enumerate(loader):
                global_step += 1

                X, Beta, SibBeta, Sij = X.to(cu.get_device()), Beta.to(cu.get_device()),\
                     SibBeta.to(cu.get_device()), Sij.to(cu.get_device(), dtype=torch.int64)

                sel_nonzero = lambda t, sij : torch.squeeze(t[torch.nonzero(sij)])
                
                tgt_beta = torch.vstack([
                    torch.mean(sel_nonzero(SibBeta[entry], Sij[entry]), dim=0) for entry in range(X.size(0)) 
                ])

                self._optimizer.zero_grad()
                beta_preds = self._phimodel.forward(X, Beta)
                loss = self._msecri(beta_preds, tgt_beta)

                loss.backward()
                self._optimizer.step()
                tq.set_postfix({"Loss": loss.item()})
                tq.update(1)                
        

    @property
    def _def_name(self):
        return f"nnphi_{self.nn_arch}"


class SynNNPhiMinHelper(NNPhiHelper):  
    """This is the default class for BBPhi.
    This des mean Recourse of Betas

    Args:
        NNPhiHelper ([type]): [description]
    """
    def __init__(self, in_dim, out_dim, nn_arch:list, rechlpr:RecourseHelper, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nn_arch = nn_arch

        assert (in_dim, out_dim) == (dh._train._Xdim+dh._train._Betadim, dh._train._Betadim), "Why are the input and output dimensions fo NNPhi inconsistent?"

        # if u need dropouts, pass it in kwargs
        phimodel = FNN(in_dim=in_dim, out_dim=out_dim, nn_arch=nn_arch, prefix="phi")
        super(SynNNPhiMinHelper, self).__init__(phimodel, rechlpr, dh, args, kwargs)

        tu.init_weights(self._phimodel)
        self._phimodel.to(cu.get_device())
        

        self._optimizer = AdamW([
            {'params': self._phimodel.parameters()},
        ], lr=self._lr)
        
    def fit_rec_beta(self, epochs, loader:data_utils.DataLoader=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader): [description]
            epochs ([type], optional): [description]. Defaults to None. 
        """
        global_step = 0
        self._phimodel.train()
        if loader is None:
            loader = self._trn_loader
        else:
            warnings.warn("Are u sure u want to pass a custom loader?")

        for epoch in range(epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (rids, X, Beta, SibBeta, Sij, Siblosses) in enumerate(loader):
                global_step += 1

                X, Beta, SibBeta, Siblosses = X.to(cu.get_device()), Beta.to(cu.get_device()),\
                     SibBeta.to(cu.get_device()), Siblosses.to(cu.get_device())

                sel_min = lambda t, losses_i : torch.squeeze(t[torch.argmin(losses_i)])

                tgt_beta = torch.vstack([
                    sel_min(SibBeta[entry], Siblosses[entry]) for entry in range(X.size(0)) 
                ])

                self._optimizer.zero_grad()
                beta_preds = self._phimodel.forward(X, Beta)
                loss = self._msecri(beta_preds, tgt_beta)

                loss.backward()
                self._optimizer.step()
                tq.set_postfix({"Loss": loss.item()})
                tq.update(1)                
        

    @property
    def _def_name(self):
        return f"nnphimin_{self.nn_arch}"


   