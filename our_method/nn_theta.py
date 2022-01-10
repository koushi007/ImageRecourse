from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import utils.common_utils as cu
import utils.torch_utils as tu
from sklearn.metrics import accuracy_score
from torch._C import dtype
from torch.optim import SGD, AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import our_method.data_helper as ourdh
from our_method.data_helper import Data
from our_method.models import LRModel


class NNthHelper(ABC):
    def __init__(self, model:nn.Module, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.dh = dh
        self.__model_copy = None
        
        self.lr = 1e-3
        self.sw = None
        self.bsz = 16

        self.__init_kwargs(kwargs)
        self.__init_loaders()
    
    def __init_kwargs(self, kwargs:dict):
        if "lr" in kwargs.keys():
            self.lr = kwargs["lr"]
        if "summarywriter" in kwargs:
            self.sw = kwargs["summarywriter"]
        if "batch_size" in kwargs:
            self.bsz = kwargs["batch_size"]

    def __init_loaders(self):
        self.trn_loader = self._dh._train.get_loader(shuffle=True, bsz=self._batch_size)
        self.tst_loader = self._dh._test.get_loader(shuffle=False, bsz=self._batch_size)
        self.val_loader = self._dh._val.get_loader(shuffle=False, bsz=self._batch_size)

# %% properties
    @property
    def _model(self):
        return self.model
    @_model.setter
    def _model(self, value):
        self.model = value

    @property
    def _dh(self):
        return self.dh
    @_dh.setter
    def _dh(self, value):
        self.dh = value

    @property
    def _batch_size(self):
        return self.bsz

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

# %% Abstract methods delegated to my children

    @abstractproperty
    def _def_name(self):
        raise NotImplementedError()

    @abstractmethod
    def fit_data(self, loader:data_utils.DataLoader=None, trn_wts=None,
                        epochs=None, steps=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader, optional): [description]. Defaults to None.
            trn_wts ([type], optional): [description]. Defaults to None.
            epochs ([type], optional): [description]. Defaults to None.
            steps ([type], optional): [description]. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()

    @abstractmethod
    def predict_labels(self, X):
        """Predict the labels

        Args:
            X ([type]): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
        raise NotImplementedError()

    @abstractmethod
    def predict_proba(self, X):   
        """Predict the probabilities

        Raises:
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """    
        raise NotImplementedError

# %% some utilities

    def copy_model(self, *args, **kwargs):
        """Stores a copy of the model
        """
        assert self.__model_copy is None, "Why are you copying an alreasy copied model?"
        self.__model_copy = deepcopy(self._model.state_dict())
    
    def apply_copied_model(self, *args, **kwargs):
        """Loads the weights of deep copied model to the origibal model
        """
        assert self.__model_copy != None
        self.model.load_state_dict(self.__model_copy)

    def clear_copied_model(self, *args, **kwargs):
        """[summary]
        """
        assert self.__model_copy is not None, "Why are you clearing an already cleared copy?"
        self.__model_copy = None

    def accuracy(self, X_test, y_test, *args, **kwargs) -> float:
        """Returns the Accuracy of predictions

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]

        Returns:
            Accuracy
        """
        return accuracy_score(y_test, self.predict_labels(X_test, *args, **kwargs))

    def grp_accuracy(self, X_test=None, y_test=None, Beta_test=None, *args, **kwargs) -> dict:
        """Computes the accuracy on a per group basis

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]
            Beta_test ([type]): [description]

        Returns:
            dict: [description]
        """
        if X_test is None:
            X_test, y_test, Beta_test = self._tst_data._X, self._tst_data._y, self._tst_data._Beta
        beta_dim = self._trn_data._Betadim
        res_dict = {}
        for beta_id in range(beta_dim):
            beta_samples = np.where(Beta_test[:, beta_id] == 1)
            res_dict[beta_id] = self.accuracy(X_test[beta_samples], y_test[beta_samples])
        return res_dict


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
        return self._xecri(T(probs).to(cu.get_device()), y_test).cpu().numpy()

    def save_model_defname(self, suffix=""):
        dir = self._def_dir
        dir.mkdir(exist_ok=True, parents=True)
        fname = dir / f"{self._def_name}{suffix}.pt"
        torch.save(self._model.state_dict(), fname)
    
    def load_model_defname(self, suffix=""):
        fname = self._def_dir / f"{self._def_name}{suffix}.pt"
        print(f"Loaded model from {str(fname)}")
        self._model.load_state_dict(torch.load(fname, map_location=cu.get_device()))

class LRNNthHepler(NNthHelper):  
    def __init__(self, in_dim, n_classes, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        self.in_dim = in_dim
        self.n_classes = n_classes
        model = LRModel(in_dim=in_dim, n_classes=n_classes, args=args, kwargs=kwargs)
        super(LRNNthHepler, self).__init__(model, dh, args, kwargs)

        tu.init_weights(self._model)
        self._model.to(cu.get_device())
        

        self._optimizer = AdamW([
            {'params': self._model.parameters()},
        ], lr=self._lr)
        
    def fit_data(self, loader:data_utils.DataLoader=None, trn_wts=None,
                        epochs=None, steps=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader): [description]
            trn_wts ([type], optional): [description]. Defaults to None. weights to be associated with each traning sample.
            epochs ([type], optional): [description]. Defaults to None. 
            steps ([type], optional): [description]. Defaults to None.
        """
        assert not(epochs is not None and steps is not None), "We will run either the specified SGD steps or specified epochs over data. We cannot run both"
        assert not(epochs is None and steps is None), "We need atleast one of steps or epochs to be specified"

        global_step = 0
        total_sgd_steps = np.inf
        total_epochs = 10
        if steps is not None:
            total_sgd_steps = steps
        if epochs is not None:
            total_epochs = epochs

        self._model.train()

        if loader is None:
            loader = self._trn_loader

        if trn_wts is None:
            trn_wts = np.ones(len(loader.dataset))
        assert len(trn_wts) == len(loader.dataset), "Pass all weights. If you intend not to train on an example, then pass the weight as 0"
        trn_wts = torch.Tensor(trn_wts).to(cu.get_device())
    
        def do_post_fit():
            # print(f"Accuracy: {self.accuracy()}")
            return

        for epoch in range(total_epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (batch_ids, batch_zids, x, y, z, beta) in enumerate(loader):
                global_step += 1
                if global_step == total_sgd_steps:
                    do_post_fit()
                    return

                x, y, batch_ids = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), batch_ids.to(cu.get_device(), dtype=torch.int64)
                self._optimizer.zero_grad()
                
                cls_out = self._model.forward_proba(x)
                loss = self._xecri_perex(cls_out, y)
                loss = torch.dot(loss, trn_wts[batch_ids]) / torch.sum(trn_wts[batch_ids])
                
                loss.backward()
                self._optimizer.step()

                tq.set_postfix({"Loss": loss.item()})
                tq.update(1)

    def accuracy(self, X_test = None, y_test=None, *args, **kwargs) -> float:
        self._model.eval()
        if X_test is None:
            X_test = self._tst_data._X
            y_test = self._tst_data._y
        return super().accuracy(X_test, y_test, *args, **kwargs)

    def predict_labels(self, X):
        self._model.eval()
        with torch.no_grad():
            X = torch.Tensor(X).to(cu.get_device())
            return self._model.forward_labels(X).cpu().numpy()
    
    def predict_proba(self, X):
        self._model.eval()
        with torch.no_grad():
            X = torch.Tensor(X).to(cu.get_device())
            return self._model.forward_proba(X).cpu().numpy()

    @property
    def _def_name(self):
        return "nntheta_lr"
