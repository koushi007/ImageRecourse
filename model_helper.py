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

    def _KLCriterion(self, reduction=None):
        if reduction is None:
            reduction = "batchmean"
        return nn.KLDivLoss(reduction=reduction)

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

class LRHelper:
    def __init__(self, trn_data, tst_data, dh, max_iter=100, *args, **kwargs) -> None:
        super().__init__(trn_data, tst_data, dh, *args, **kwargs)
        self.clf = LogisticRegression(penalty="none", random_state=42, max_iter=max_iter)
        self.weights = None
        self._model = self.clf
        self._trn_data = trn_data
        self._tst_data = tst_data

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

    def accuracy(self, X_test, y_test, *args, **kwargs) -> float:
        return sklearn.metrics.accuracy_score(y_test, self.predict(X_test, *args, **kwargs))

    def grp_accuracy(self, X: np.array, Beta: np.array, y: np.arange, *args, **kwargs) -> dict:  
        beta_dim = len(Beta[0])
        res_dict = {}
        for beta_id in range(beta_dim):
            beta_samples = np.where(Beta[:, beta_id] == 1)
            res_dict[beta_id] = self.accuracy(X[beta_samples], y[beta_samples])
        return res_dict

class SynRecourse(ModelHelper, ABC):
    """
    This is a base class for all models that provide recourse to our Synthetic dataset.
    Donot instantiate objects out of tjis class.
    """
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

        self.rec_optimizer = AdamW([
            {'params': self.rec_model.parameters()},
        ], lr=self._lr)

        self.cls_optimizer = AdamW([
            {'params': self._model.parameters()},
        ], lr=self._lr)

        self.def_dir = None
        self.def_name = None


    @abstractmethod
    def fit_epoch(self, epoch, loader=None, *args, **kwargs):
        raise NotImplementedError()


    def entropy_bias(self, loader=None, epochs=5):
        self.rec_model.train()
        if loader is None:
            loader = self.trn_loader

        step = 0
        target = torch.tensor(torch.ones(self.batch_size, self._Xdim)).to(cu.get_device())
        target = target / self._Xdim

        for epoch in range(epochs):
            tq = tqdm(total=len(loader), desc="Loss")
            for id, x, _, _, beta in loader:
                step += 1
                x, beta = x.to(cu.get_device()), beta.to(cu.get_device())
                self.rec_optimizer.zero_grad()

                y_preds = self.rec_model(x, beta)
                y_preds = torch.softmax(y_preds, dim=1)
                y_preds = torch.log(y_preds)

                loss = self._KLCriterion(y_preds, target)
                loss.backward()
                self.rec_optimizer.step()
                self._sw.add_scalar("Ent-Prior", loss.item(), step)
                tq.set_description(f"Lossh: {loss.item()}")
                tq.update(1)

    def accuracy(self, X_test = None, y_test=None, *args, **kwargs) -> float:
        if X_test is None:
            X_test = self._tst_data._X
            y_test = self._tst_data._0INDy
            kwargs["Beta"] = self._tst_data._Beta
        return super().accuracy(X_test, y_test, *args, **kwargs)

    def grp_accuracy(self, X_test: np.array = None, Beta_test: np.array = None, y_test: np.arange=None, *args, **kwargs) -> dict:
        if X_test is None:
            X_test = self._tst_data._X
            y_test = self._tst_data._0INDy
            Beta_test = self._tst_data._Beta
        res_dict = {}
        for beta_id in range(self._Betadim):
            beta_samples = np.where(Beta_test[:, beta_id] == 1)
            kwargs = {"Beta": Beta_test[beta_samples]}
            res_dict[beta_id] = self.accuracy(X_test[beta_samples], y_test[beta_samples], **kwargs)
        return res_dict

    def recourse_accuracy(self):
        loader = self.tst_loader

        self.rec_model.eval()
        sum_acc = 0
        with torch.no_grad():
            for id, x, y, z, beta in loader:
                x, y, z, beta = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), z.to(cu.get_device()), beta.to(cu.get_device())
                rec_beta = self.rec_model(x, beta)
                rec_beta = rec_beta > 0.5
                x_rec = torch.mul(z, rec_beta)
                acc = self.accuracy(x_rec.cpu().numpy(), y.cpu().numpy(), Beta=rec_beta.cpu().numpy())
                sum_acc += acc
        return sum_acc / len(loader)

    
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

    @property
    def _def_dir(self):
        return self.def_dir
    @_def_dir.setter
    def _def_dir(self, value):
        self.def_dir = value
    
    @property
    def _def_name(self):
        return self.def_name
    @_def_name.setter
    def _def_name(self, value):
        self.def_name = value
    
    def save_model_defname(self, suffix = ""):
        state_dict = {
            "recourse": self.rec_model.state_dict(),
            "classifier": self._model.state_dict()
        }
        dir = self._def_dir
        dir.mkdir(exist_ok=True, parents=True)
        torch.save(state_dict, self._def_dir / (self._def_name + suffix + ".pt"))
    
    def load_model_defname(self, suffix=""):
        state_dict = torch.load(self._def_dir / (self._def_name + suffix + ".pt"), map_location=cu.get_device())
        self._model.load_state_dict(state_dict["classifier"])
        self.rec_model.load_state_dict(state_dict["recourse"])
        print(f"Models loader from {str(self._def_dir / (self._def_name + suffix + '.pt'))}")

    def load_def_classifier(self, suffix=""):
        fname = f"results/models/nn_cls/{str(self.dh)}/classifier{suffix}.pt"
        print(f"Loaded NN_cls classifier from {str(fname)}")
        self._model.load_state_dict(torch.load(fname, map_location=cu.get_device()))

    def save_rec_model(self, suffix=""):
        fname = self._def_dir / f"rec_model{suffix}.pt"
        torch.save(self.rec_model.state_dict(), fname)
    
    def load_rec_model(self, suffix=""):
        fname = self._def_dir / f"rec_model{suffix}.pt"
        self.rec_model.load_state_dict(torch.load(fname, map_location=cu.get_device()))

class BaselineHelper(SynRecourse):
    def __init__(self, trn_data, tst_data, dh, nn_arch=[10, 10], *args, **kwargs) -> None:
        super(BaselineHelper, self).__init__(trn_data, tst_data, dh, nn_arch, *args, **kwargs)

        self._def_dir = Path(f"results/models/baseline/{str(self.dh)}")
        self._def_name = "baseline"

    def fit_epoch(self, epoch, loader=None, *args, **kwargs):
        
        inter_iters = -1
        if "interleave_iters" in kwargs:
            inter_iters = kwargs["interleave_iters"]

        self._model.train()
        self.rec_model.train()

        if loader is None:
            loader = self.grp_trn_loader
        
        global_step = epoch * len(loader)
        tq = tqdm(total=len(loader), desc="Loss")

        num_grp = self.dh._train.B_per_i
        
        for local_step, (id_grp, X_grp, y_grp, Z_grp, Beta_grp) in enumerate(loader):
            util_grp = 0
            for id, x, y, z, beta in zip(id_grp, X_grp, y_grp, Z_grp, Beta_grp):
                
                x, y, beta = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), beta.to(cu.get_device())
                self._optimizer.zero_grad()
                
                # this is P(\beta_ir | ij)
                rec_beta = self.rec_model.forward(x, beta)
                rec_beta = torch.sigmoid(rec_beta)

                rec_beta_agg = rec_beta.repeat_interleave(num_grp, dim=0)
                beta_agg = beta.repeat(num_grp, 1)
                rec_beta_loss = torch.mul(beta_agg, torch.log(rec_beta_agg+1e-5)) + torch.mul(1-beta_agg, torch.log(1-rec_beta_agg+1e-5))
                rec_beta_loss = torch.sum(rec_beta_loss, dim=1)

                cls_out = self._model.forward(x, beta)
                cls_out = torch.softmax(cls_out, dim=1)
                cls_out = torch.gather(cls_out, 1, y.view(-1, 1)).squeeze()
                cls_out = torch.log(cls_out + 1e-5)
                cls_loss = cls_out.repeat(num_grp)

                if inter_iters != -1:
                    if (local_step % (2*inter_iters)) % inter_iters == 0:
                        do_rec, do_cls = 0, 1
                    else:
                        do_rec, do_cls = 1, 0
                else:
                    do_rec, do_cls = 1, 1
                util = (do_rec * rec_beta_loss) + (do_cls * cls_loss)

                util = util.view(num_grp, -1)
                util, max_idxs = torch.max(util, dim=0)
                util = torch.sum(util)
                
                util_grp += util

            cls_loss_sw = torch.gather(cls_loss.view(num_grp, -1), 0, max_idxs.view(-1, 1))
            rec_loss_sw = torch.gather(rec_beta_loss.view(num_grp, -1), 0, max_idxs.view(-1, 1))
            self._sw.add_scalar("cls_loss", torch.mean(cls_loss_sw), global_step+local_step)
            self._sw.add_scalar("rec_loss", torch.mean(rec_loss_sw), global_step+local_step)

            loss = -util_grp/len(X_grp)
            loss.backward()
            self._optimizer.step()
            self._sw.add_scalar("Loss", loss.item(), global_step+local_step)
            tq.set_description(f"Loss: {loss.item()}")
            tq.update(1)

class BaselineKLHelper(SynRecourse):
    def __init__(self, trn_data, tst_data, dh, nn_arch=[10, 10], *args, **kwargs) -> None:
        super(BaselineKLHelper, self).__init__(trn_data, tst_data, dh, nn_arch, *args, **kwargs)

        self._def_dir = Path(f"results/models/klbaseline/{str(self.dh)}")
        self._def_name = "baselinekl"

    def fit_epoch(self, epoch, loader=None, *args, **kwargs):
        
        inter_iters = -1
        if "interleave_iters" in kwargs:
            inter_iters = kwargs["interleave_iters"]

        self._model.train()
        self.rec_model.train()

        if loader is None:
            loader = self.grp_trn_loader
        
        global_step = epoch * len(loader)
        tq = tqdm(total=len(loader), desc="Loss")

        num_grp = self.dh._train.B_per_i
        uni_kl_targets = torch.ones(num_grp*num_grp, self._Betadim) / self._Betadim
        uni_kl_targets = uni_kl_targets.to(cu.get_device())
        ent_decay = torch.tensor([0.9]).to(cu.get_device())
    
        for local_step, (id_grp, X_grp, y_grp, Z_grp, Beta_grp) in enumerate(loader):
            util_grp = 0
            for id, x, y, z, beta in zip(id_grp, X_grp, y_grp, Z_grp, Beta_grp):
                
                x, y, beta = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), beta.to(cu.get_device())
                self._optimizer.zero_grad()
                
                # this is P(\beta_ir | ij)
                rec_beta = self.rec_model.forward(x, beta)
                rec_beta = torch.softmax(rec_beta, dim=1)
                rec_beta = torch.log(rec_beta + 1e-5)

                rec_beta_agg = rec_beta.repeat_interleave(num_grp, dim=0)
                beta_agg = beta.repeat(num_grp, 1)

                rec_beta_loss_ir = -self._KLCriterion(reduction="none")(rec_beta_agg, beta_agg / torch.sum(beta_agg, dim=1).view(-1, 1))
                rec_beta_loss_uni = self._KLCriterion(reduction="none")(rec_beta_agg, uni_kl_targets)


                rec_beta_loss_ir = torch.sum(rec_beta_loss_ir, dim=1)
                rec_beta_loss_uni = torch.sum(rec_beta_loss_uni, dim=1)
                self._sw.add_scalar("-KL(betair || beta|xij)", torch.mean(rec_beta_loss_ir).item(), global_step+local_step)
                self._sw.add_scalar("KL(betair|| uni)", torch.mean(rec_beta_loss_uni).item(), global_step+local_step)
                rec_beta_loss = rec_beta_loss_ir + torch.pow(ent_decay, epoch+1) * rec_beta_loss_uni

                cls_out = self._model.forward(x, beta)
                cls_out = torch.softmax(cls_out, dim=1)
                cls_out = torch.gather(cls_out, 1, y.view(-1, 1)).squeeze()
                cls_out = torch.log(cls_out + 1e-5)
                cls_loss = cls_out.repeat(num_grp)

                if inter_iters != -1:
                    if (local_step % (2*inter_iters)) % inter_iters == 0:
                        do_rec, do_cls = 0, 1
                    else:
                        do_rec, do_cls = 1, 0
                else:
                    do_rec, do_cls = 1, 1
                util = (do_rec * rec_beta_loss) + (do_cls * cls_loss)

                util = util.view(num_grp, -1)
                util, max_idxs = torch.max(util, dim=0)
                util = torch.sum(util)
                
                util_grp += util

                cls_loss_sw = torch.gather(cls_loss.view(num_grp, -1), 0, max_idxs.view(-1, 1))
                rec_loss_sw = torch.gather(rec_beta_loss.view(num_grp, -1), 0, max_idxs.view(-1, 1))
                self._sw.add_scalar("cls_loss", torch.mean(cls_loss_sw), global_step+local_step)
                self._sw.add_scalar("rec_loss", torch.mean(rec_loss_sw), global_step+local_step)


            loss = -util_grp/len(X_grp)
            loss.backward()
            self._optimizer.step()
            self._sw.add_scalar("Loss", loss.item(), global_step+local_step)
            tq.set_description(f"Loss: {loss.item()}")
            tq.update(1)

class Method1Helper(SynRecourse):
    def __init__(self, trn_data, tst_data, dh, nn_arch=[10, 10], *args, **kwargs) -> None:
        super().__init__(trn_data, tst_data, dh, nn_arch=nn_arch, *args, **kwargs)

        self._def_dir = Path(f"results/models/method1/{str(self.dh)}")
        self._def_name = "method1"

    def fit_epoch(self, epoch, loader=None, *args, **kwargs):

        print("Training Method 1")
        
        self._model.train()

        if loader is None:
            loader = self.grp_trn_loader
        
        global_step = epoch * len(loader)
        tq = tqdm(total=len(loader), desc="Loss")

        num_grp = self.dh._train.B_per_i
    
        for local_step, (id_grp, X_grp, y_grp, Z_grp, Beta_grp) in enumerate(loader):
            util_grp = 0
            for id, x, y, z, beta in zip(id_grp, X_grp, y_grp, Z_grp, Beta_grp):
                
                x, y, beta = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), beta.to(cu.get_device())
                self._optimizer.zero_grad()
                
                cls_out = self._model.forward(x, beta)
                cls_out = torch.softmax(cls_out, dim=1)
                cls_out = torch.gather(cls_out, 1, y.view(-1, 1)).squeeze()
                cls_out = torch.log(cls_out + 1e-5)

                util, max_idxs = torch.max(cls_out, dim=0)
                util = torch.sum(util)

                if beta[max_idxs][0] != 1 and epoch > 5:
                    # print(f"Found {beta}")
                    # print(torch.exp(cls_out))
                    continue
                    pass
                
                util_grp += util

            loss = -util_grp/len(X_grp)
            loss.backward()
            self._optimizer.step()
            self._sw.add_scalar("Loss", loss.item(), global_step+local_step)
            tq.set_description(f"Loss: {loss.item()}")
            tq.update(1)

class NNHelper(SynRecourse):
    def __init__(self, trn_data, tst_data, dh, nn_arch=[10, 10], *args, **kwargs) -> None:
        super().__init__(trn_data, tst_data, dh, nn_arch, *args, **kwargs)

        self._def_dir = Path(f"results/models/nn_cls/{str(self.dh)}")
        self._def_name = "classifier"

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