import numpy as np
from abc import ABC, abstractmethod, abstractproperty
from numpy.lib.function_base import bartlett
from numpy.lib.npyio import load
from sklearn.linear_model import LogisticRegression
import sklearn
from torch._C import dtype
from torch.types import Number
from FNN import FNN
from data_helper import Data, DataHelper, SyntheticDataHelper
import utils.common_utils as cu
import torch
import torch.utils.data as data_utils
import torch.nn as nn
from copy import deepcopy
from torch.optim import SGD, AdamW
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm

class ModelHelper(ABC):
    def __init__(self, trn_data:Data, tst_data:Data, dh:DataHelper, *args, **kwargs) -> None:
        super().__init__()

        self.trn_data = trn_data
        self.tst_data = tst_data
        self.dh = dh

        self.model = None
        self.trn_loader = None
        self.tst_loader = None
        self.trngrp_loader = None
        self.criterion = None
        self.optimizer = None
        self.sw = None

        self.batch_size = 16
        self.lr = 1e-3

        self.__init_kwargs(kwargs)
        self.__init_loaders()
    
    def __init_kwargs(self, kwargs):
        if "batch_size" in kwargs:
            self.batch_size = kwargs["batch_size"]   
        if "summarywriter" in kwargs:
            self.sw = kwargs["summarywriter"]

    def __init_loaders(self):
        self.trn_loader = self._trn_data.get_loader(shuffle=True, bsz=self.batch_size)
        self.grp_trn_loader = self._trn_data.get_grp_loader(shuffle=True, bsz=self.batch_size)
        self.tst_loader = self._tst_data.get_loader(shuffle=False, bsz=self.batch_size)

    @property
    def _trn_data(self) -> Data:
        return self.trn_data
    @_trn_data.setter
    def _trn_data(self, value):
        raise ValueError("Why are u setting the data object once again?")   

    @property
    def _tst_data(self) -> Data:
        return self.tst_data
    @_tst_data.setter
    def _tst_data(self, value):
        raise ValueError("Why are u setting the data object once again?")   

    @property
    def _device(self) -> str:
        return cu.get_device()

    @property
    def _model(self) -> nn.Module:
        if self.model is None:
            raise ValueError("Model is not ste yet!")
        return self.model
    @_model.setter
    def _model(self, value):
        self.model = value

    @property
    def _Xdim(self):
        return len(self._trn_data._X[0])
    @property
    def _Betadim(self):
        return len(self._trn_data._Beta[0])
    @property
    def _nclasses(self):
        return self._trn_data._num_classes


    @property
    def _criterion(self):
        if  self.criterion == None:
            raise ValueError("Criterion not yet set")
        return self.criterion
    @_criterion.setter
    def _criterion(self, value):
        self.criterion = value

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

    def get_defdir(self):
        return Path("results/models")



    @abstractmethod
    def predict(self, X, *args, **kwargs):
        """Predicts the class labels for given X

        Args:
            X ([type]): [description]
        """
        pass

    @abstractmethod
    def predict_proba(self, X, *args, **kwargs):
        """Predicts the logits for X

        Args:
            X ([type]): [description]
        """
        pass

    def accuracy(self, X_test, y_test, *args, **kwargs) -> float:
        """Returns the Accuracy of predictions

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]

        Returns:
            Accuracy
        """
        return sklearn.metrics.accuracy_score(y_test, self.predict(X_test, *args, **kwargs))

    @abstractmethod
    def grp_accuracy(self, X:np.array, Beta:np.array, y:np.arange, *args, **kwargs) -> dict:
        """Returnms the group wise accuracy for each beta as a dict

        Args:
            X ([type]): [description]
            Beta ([type]): [description]
            y ([type]): [description]

        Raises:
            NotImplementedError: [description]
        Returns:
            Dictionary for each beta dimension
        """
        raise NotImplementedError()

class LRHelper(ModelHelper):
    def __init__(self, trn_data, tst_data, dh, max_iter=100, *args, **kwargs) -> None:
        super().__init__(trn_data, tst_data, dh, *args, **kwargs)
        self.clf = LogisticRegression(penalty="none", random_state=42, max_iter=max_iter)
        self.weights = None
        self._model = self.clf

    @property
    def _weights(self) -> np.array:
        assert self.weights is not None, "Access the weights only after training the model"
        return self.weights
    @_weights.setter
    def _weights(self, value):
        self.weights = value

    def fit(self, X=None, y=None, *args, **kwargs):
        if X is None:
            assert y is None, "Only One of X, y cannot be none"
            X = self._trn_data._X
            y = self._trn_data._0INDy
        self.clf.fit(X, y)
        self._weights = np.squeeze(self.clf.coef_)
    
    def predict(self, X, *args, **kwargs):

        return self.clf.predict(X)

    def predict_proba(self, X, *args, **kwargs):
        return self.clf.predict_proba(X)

    def grp_accuracy(self, X: np.array, Beta: np.array, y: np.arange, *args, **kwargs) -> dict:  
        beta_dim = len(Beta[0])
        res_dict = {}
        for beta_id in range(beta_dim):
            beta_samples = np.where(Beta[:, beta_id] == 1)
            res_dict[beta_id] = self.accuracy(X[beta_samples], y[beta_samples])
        return res_dict


class RecourseHelper(ModelHelper):
    def __init__(self, trn_data, tst_data, dh, nn_arch=[10, 10], *args, **kwargs) -> None:
        super().__init__(trn_data, tst_data, dh, *args, **kwargs)
        
        self.cls_nn_arch:list = deepcopy(nn_arch)
        self.cls_nn_arch.append(self._nclasses)
        kwargs = {
            "prefix": "classifier_"
        }
        self._model = FNN(in_dim=self._Xdim+self._Betadim, nn_arch=self.cls_nn_arch, **kwargs)
        self._model.to(cu.get_device())
        cu.init_weights(self._model)

        self.rec_nn_arch = deepcopy(nn_arch)
        self.rec_nn_arch.append(self._Betadim)
        kwargs = {
            "prefix": "recourse_"
        }
        self.rec_model = FNN(in_dim=self._Xdim+self._Betadim, nn_arch=self.rec_nn_arch, **kwargs)
        self.rec_model.to(cu.get_device())
        cu.init_weights(self.rec_model)

        self._optimizer = AdamW([
            {'params': self._model.parameters()},
            {'params': self.rec_model.parameters()},
        ], lr=self._lr)


    def fit_epoch(self, epoch, loader=None):
        self._model.train()
        self.rec_model.train()

        if loader is None:
            loader = self.grp_trn_loader
        
        step = epoch * len(loader)
        tq = tqdm(total=len(loader), desc="Loss")
        for id_grp, X_grp, y_grp, Z_grp, Beta_grp in loader:
            step += 1
            self._optimizer.zero_grad()

            util_grp = 0
            for id, x, y, z, beta in zip(id_grp, X_grp, y_grp, Z_grp, Beta_grp):
                x, y, beta = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), beta.to(cu.get_device())
                
                num_grp = x.size(0)
                rec_out = self.rec_model.forward(x, beta)
                rec_out = torch.sigmoid(rec_out)

                rec_out_agg = rec_out.repeat_interleave(num_grp, dim=0)
                beta_agg = beta.repeat(num_grp, 1)
                rec_out_agg = torch.mul(beta_agg, torch.log(rec_out_agg + 1e-5)) + torch.mul(1-beta_agg, torch.log(1-rec_out_agg+1e-5))
                rec_out_agg = torch.sum(rec_out_agg, dim=1)

                cls_out = self._model.forward(x, beta)
                cls_out = torch.softmax(cls_out, dim=1)
                cls_out = torch.gather(cls_out, 1, y.view(-1, 1)).squeeze()
                cls_out = torch.log(cls_out)
                cls_out_agg = cls_out.repeat_interleave(num_grp, dim=0)

                util = rec_out_agg + cls_out_agg
                util = util.view(num_grp, -1)
                util, _ = torch.max(util, dim=0)
                util = torch.sum(util)

                util_grp += util

            loss = -util_grp/len(X_grp)
            loss.backward()
            self._optimizer.step()
            self._sw.add_scalar("Loss", loss.item(), step)
            tq.set_description(f"Loss: {loss.item()}")
            tq.update(1)

    

    def accuracy(self, X_test, y_test, *args, **kwargs) -> float:
        return super().accuracy(X_test, y_test, *args, **kwargs)

    def grp_accuracy(self, X: np.array, Beta: np.array, y: np.arange, *args, **kwargs) -> dict:
        res_dict = {}
        for beta_id in range(self._Betadim):
            beta_samples = np.where(Beta[:, beta_id] == 1)
            kwargs = {"Beta": Beta[beta_samples]}
            res_dict[beta_id] = self.accuracy(X[beta_samples], y[beta_samples], **kwargs)
        return res_dict
    
    def predict(self, X, *args, **kwargs) -> np.array:
        self._model.eval()
        Beta = kwargs["Beta"]
        X, Beta = torch.Tensor(X).to(cu.get_device()), torch.Tensor(Beta).to(cu.get_device())
        with torch.no_grad():
            y_preds = self._model.forward(X, Beta)
        y_preds = torch.softmax(y_preds, dim=1)
        y_preds = torch.argmax(y_preds, dim=1)
        return y_preds.cpu().numpy()

    def predict_proba(self, X, *args, **kwargs) -> np.array:
        self._model.eval()
        Beta = kwargs["Beta"]
        X, Beta = torch.Tensor(X).to(cu.get_device()), torch.Tensor(Beta).to(cu.get_device())
        with torch.no_grad():
            y_preds = self._model.forward(X, Beta)
        y_preds = torch.softmax(y_preds, dim=1)
        return y_preds.cpu().numpy()

    def predict_betas(self, X, Beta) -> np.array:
        self.rec_model.eval()
        X, Beta = torch.Tensor(X).to(cu.get_device()), torch.Tensor(Beta).to(cu.get_device())
        with torch.no_grad():
            pred_beta = self.rec_model.forward(X, Beta)
            pred_beta = torch.sigmoid(pred_beta)
        return pred_beta.cpu().numpy()

    def get_defdir(self):
        return super().get_defdir() / "recourse" / str(self.dh)
    
    def get_defname(self):
        return "recourse_theta_phi.pt"
    
    def save_model_def(self):
        state_dict = {
            "recourse": self.rec_model.state_dict(),
            "classifier": self._model.state_dict()
        }
        dir = self.get_defdir()
        dir.mkdir(exist_ok=True, parents=True)
        torch.save(state_dict, self.get_defdir() / self.get_defname())
    
    def load_model_defname(self):
        state_dict = torch.load(self.get_defdir() / self.get_defname(), map_location=cu.get_device())
        self._model.load_state_dict(state_dict["classifier"])
        self.rec_model.load_state_dict(state_dict["recourse"])
        print(f"Models loader from {str(self.get_defdir() / self.get_defname())}")



class NNHelper(ModelHelper):
    def __init__(self, trn_data, tst_data, dh, nn_arch=[10, 10], *args, **kwargs) -> None:
        super().__init__(trn_data, tst_data, dh, *args, **kwargs)
        
        self.cls_nn_arch:list = deepcopy(nn_arch)
        self.cls_nn_arch.append(self._nclasses)
        kwargs = {
            "prefix": "classifier_"
        }
        self._model = FNN(in_dim=self._Xdim+self._Betadim, nn_arch=self.cls_nn_arch, **kwargs)
        self._model.to(cu.get_device())
        cu.init_weights(self._model)

        self._optimizer = AdamW([
            {'params': self._model.parameters()},
        ], lr=self._lr)
        self._criterion = nn.CrossEntropyLoss()


    def fit_epoch(self, epoch, loader=None):
        self._model.train()

        if loader is None:
            loader = self.trn_loader
        
        step = epoch * len(loader)
        tq = tqdm(total=len(loader), desc="Loss")
        for id, x, y, z, beta in loader:
            step += 1
            self._optimizer.zero_grad()

            x, y, beta = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), beta.to(cu.get_device())

            cls_out = self._model.forward(x, beta)
            cls_out = torch.softmax(cls_out, dim=1)
        
            loss = self._criterion(cls_out, y)
            loss.backward()
            self._optimizer.step()
            self._sw.add_scalar("Loss", loss.item(), step)
            tq.set_description(f"Loss: {loss.item()}")
            tq.update(1)
    
    def accuracy(self, X_test, y_test, *args, **kwargs) -> float:
        return super().accuracy(X_test, y_test, *args, **kwargs)

    def grp_accuracy(self, X: np.array, Beta: np.array, y: np.arange, *args, **kwargs) -> dict:
        res_dict = {}
        for beta_id in range(self._Betadim):
            beta_samples = np.where(Beta[:, beta_id] == 1)
            kwargs = {"Beta": Beta[beta_samples]}
            res_dict[beta_id] = self.accuracy(X[beta_samples], y[beta_samples], **kwargs)
        return res_dict
    
    def predict(self, X, *args, **kwargs) -> np.array:
        self._model.eval()
        Beta = kwargs["Beta"]
        X, Beta = torch.Tensor(X).to(cu.get_device()), torch.Tensor(Beta).to(cu.get_device())
        with torch.no_grad():
            y_preds = self._model.forward(X, Beta)
        y_preds = torch.softmax(y_preds, dim=1)
        y_preds = torch.argmax(y_preds, dim=1)
        return y_preds.cpu().numpy()

    def predict_proba(self, X, *args, **kwargs) -> np.array:
        self._model.eval()
        Beta = kwargs["Beta"]
        X, Beta = torch.Tensor(X).to(cu.get_device()), torch.Tensor(Beta).to(cu.get_device())
        with torch.no_grad():
            y_preds = self._model.forward(X, Beta)
        y_preds = torch.softmax(y_preds, dim=1)
        return y_preds.cpu().numpy()

    def get_defdir(self):
        return super().get_defdir() / "nn_cls" / str(self.dh)
    
    def get_defname(self):
        return "classifier.pt"
    
    def save_model_def(self):
        dir = self.get_defdir()
        dir.mkdir(exist_ok=True, parents=True)
        torch.save(self._model.state_dict(), dir / self.get_defname())
    
    def load_model_defname(self):
        state_dict = torch.load(self.get_defdir() / self.get_defname(), map_location=cu.get_device())
        self._model.load_state_dict(state_dict)
        print(f"Models loader from {str(self.get_defdir() / self.get_defname())}")



