from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from pathlib import Path
#from statistics import quantiles

import numpy as np
import sklearn
from sympy import isprime
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
from our_method.models import LRModel, ResNET
import our_method.constants as constants


class NNthHelper(ABC):
    def __init__(self, model:nn.Module, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.dh = dh
        self.__model_copy = None
        
        self.lr = 1e-3
        self.sw = None
        self.batch_size = 16
        self.momentum = 0
        self.lr_scheduler = None

        self.__init_kwargs(kwargs)
        self.__init_loaders()
    
    def __init_kwargs(self, kwargs:dict):

        self.kwargs = kwargs

        if constants.LRN_RATTE in kwargs.keys():
            self.lr = kwargs[constants.LRN_RATTE]
        if constants.SW in kwargs.keys():
            self.sw = kwargs[constants.SW]
        if constants.BATCH_SIZE in kwargs.keys():
            self.batch_size = kwargs[constants.BATCH_SIZE]
        if constants.MOMENTUM in kwargs.keys():
            self.momentum = kwargs[constants.MOMENTUM]

    def __init_loaders(self):
        self.trn_loader = self._dh._train.get_loader(shuffle=True, batch_size=self._batch_size)
        self.tst_loader = self._dh._test.get_loader(shuffle=False, batch_size=128)
        self.val_loader = self._dh._val.get_loader(shuffle=False, batch_size=128)

# %% properties
    @property
    def _model(self) -> nn.Module:
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
    def _optimizer(self) -> optim.Optimizer:
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
        return Path("/mnt/infonas/data/nlokesh/image_recourse/our_method/results/models/nn_theta")

    @property
    def _momentum(self):
        return self.momentum

    @property
    def _lr_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        return self.lr_scheduler
    @_lr_scheduler.setter
    def _lr_scheduler(self, value):
        self.lr_scheduler = value

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

    def predict_labels(self, X):
        self._model.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.Tensor(X).to(cu.get_device())
            X = X.to(cu.get_device())
            return self._model.forward_labels(X).cpu().numpy()
    
    def predict_proba(self, X):
        self._model.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.Tensor(X)
            X = X.to(cu.get_device())
            return self._model.forward_proba(X).cpu().numpy()

    def accuracy(self, X_test = None, y_test=None, loader=None, *args, **kwargs) -> float:
        self._model.eval()
        if X_test is not None:
            return accuracy_score(y_test, self.predict_labels(X_test, *args, **kwargs))
        else:
            if loader is None:
                loader = self._tst_loader
            accs = []
            for _, _, x, y, _, _ in loader:
                accs.append(accuracy_score(y.cpu().numpy(), self.predict_labels(x)) * len(x))
            return np.sum(accs) / len(loader.dataset)


    def grp_accuracy(self, loader:data_utils.DataLoader, *args, **kwargs) -> dict:
        """Computes the accuracy on a per group basis

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]
            Beta_test ([type]): [description]

        Returns:
            dict: [description]
        """
        if loader is not None:
            raise NotImplementedError()

        loader = self._tst_loader
        Beta_test = self._dh._test._Beta
        res_dict = {}
        beta_dim = self._trn_data._Betadim
        res_dict = {}
        for beta_id in range(beta_dim):
            beta_values = set(Beta_test[:, beta_id])
            for beta_v in beta_values:
                beta_samples = np.where(Beta_test[:, beta_id] == beta_v)[0]
                beta_value_loader = tu.get_loader_subset(loader, beta_samples)
                res_dict[f"id-{beta_id}:val-{beta_v}"] = self.accuracy(loader=beta_value_loader)
        return res_dict

    def beta_accuracy(self, loader:data_utils.DataLoader,*args, **kwargs) -> dict:
        """Computes the accuracy on a per group basis

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]
            Beta_test ([type]): [description]

        Returns:
            dict: [description]
        """
        if loader is not None:
            raise NotImplementedError()

        loader = self._tst_loader
        Beta_test = self._dh._test._Beta
        y_test = self._dh._test._y
        res_dict = {}
        beta_all = [[3,1,0],[2,1,0],[0,1,0],[1,0,1],[4,2,2],[5,2,3],[3,2,1],[2,0,2],[0,0,3]]

        # beta_dim = self._trn_data._Betadim
        # res_dict = {}
        for id in range(10):
            for beta in beta_all:
                beta_samples = np.where(np.logical_and((Beta_test == beta).all(axis=1),y_test==id))[0]
                print(beta_samples.shape)
                beta_value_loader = tu.get_loader_subset(loader, beta_samples)
                res_dict[f"id-{id} beta-{beta}"] = self.accuracy(loader=beta_value_loader)
        return res_dict

    def get_trnloss_quantiles(self) -> dict:
        loader = self._dh._test.get_loader_with_ideal(False,1)
        batch_losses = lambda batchx, batchy: self.get_batchloss_perex(batchx, batchy)
        all_losses = [(float(batch_losses(x, y)),ideal_beta) for dids, ideal_beta, x, y, _, _ in loader]
        betas_sorted = [i[1] for i in sorted(all_losses,reverse=True)]
        qu = [1, 2, 3, 5, 7, 10, 20, 30, 40, 50,100]
        total_beta = len(betas_sorted)
        res_dict = {}
        for q in qu:
            betas = betas_sorted[:int((total_beta*q)/100)]
            betas_ideal = sum(betas)
            ideal_to_non_ideal = betas_ideal*1.0/int((total_beta*q)/100)
            res_dict[f"{q}"] = float(ideal_to_non_ideal)


        return res_dict

    def get_batchloss_perex(self, X_test, y_test):
        """Gets the cross entropy loss per example

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]

        Returns:
            np.array of losses
        """
        T = torch.Tensor
        if not isinstance(X_test, T):
            X_test, y_test = T(X_test), T(y_test)

        probs = self.predict_proba(X_test)
        
        return self._xecri_perex(
            T(probs).to(cu.get_device()), T(y_test).to(cu.get_device(), dtype=torch.int64)
        ).cpu().numpy()

    def get_loaderlosses_perex(self, loader=None) -> np.array:
        if loader is None:
            loader = self._dh._train.get_loader(shuffle=False, batch_size=self._batch_size)
        batch_losses = lambda batchx, batchy: self.get_batchloss_perex(batchx, batchy)
        all_losses = np.hstack([batch_losses(x, y) for dids, zids, x, y, _, _ in loader])
        return all_losses

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
        if not isinstance(X_test, T):
            X_test, y_test = T(X_test), T(y_test)
        
        probs = self.predict_proba(X_test)

        return self._xecri(
            T(probs).to(cu.get_device()), T(y_test).to(cu.get_device(), dtype=torch.int64)
        ).cpu().numpy()

    def save_model_defname(self, suffix=""):
        dir = self._def_dir
        dir.mkdir(exist_ok=True, parents=True)
        fname = dir / f"{self._def_name}{suffix}.pt"
        torch.save(self._model.state_dict(), fname)

    def save_optim_defname(self, suffix=""):
        dir = self._def_dir
        fname = dir / f"{self._def_name}{suffix}-optim.pt"
        torch.save(self._optimizer.state_dict(), fname)
    
    def load_model_defname(self, suffix=""):
        fname = self._def_dir / f"{self._def_name}{suffix}.pt"
        print(f"Loaded NN theta model from {str(fname)}")
        self._model.load_state_dict(torch.load(fname, map_location=cu.get_device()))

    def load_optim_defname(self, suffix=""):
        dir = self._def_dir
        fname = dir / f"{self._def_name}{suffix}-optim.pt"
        print(f"Loaded NN theta Optimizer from {str(fname)}")
        self._optimizer.load_state_dict(torch.load(fname, map_location=cu.get_device()))

class LRNNthHepler(NNthHelper):  
    def __init__(self, in_dim, n_classes, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        self.in_dim = in_dim
        self.n_classes = n_classes
        model = LRModel(in_dim=in_dim, n_classes=n_classes, *args, **kwargs)
        super(LRNNthHepler, self).__init__(model, dh, *args, **kwargs)

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

    @property
    def _def_name(self):
        return "lr"

class ResNETNNthHepler(NNthHelper):  
    def __init__(self, n_classes, dh:ourdh.DataHelper, *args, **kwargs) -> None:
        self.n_classes = n_classes
        model = ResNET(out_dim=n_classes, *args, **kwargs)
        super(ResNETNNthHepler, self).__init__(model, dh, *args, **kwargs)

        # For Resnet, we should never initialize weights
        self._model.to(cu.get_device())

        self._optimizer = optim.SGD(
            self._model.parameters(), lr=self._lr, momentum=self._momentum
            )
        
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

        # Initialize weights to perform average loss
        if trn_wts is None:
            trn_wts = np.ones(len(loader.dataset))
        assert len(trn_wts) == len(loader.dataset), "Pass all weights. If you intend not to train on an example, then pass the weight as 0"
        trn_wts = torch.Tensor(trn_wts).to(cu.get_device())

        if constants.SCHEDULER in kwargs.keys():
            if constants.SCHEDULER_TYPE in kwargs.keys():
                raise NotImplementedError()
            self._lr_scheduler = tu.get_lr_scheduler(self._optimizer, scheduler_name="cosine_annealing", n_rounds=epochs)
        if constants.OPTIMIZER in kwargs.keys():
            self._optimizer = kwargs["optimizer"]
        if constants.SW in kwargs.keys():
            self._sw = kwargs[constants.SW]

        for epoch in range(total_epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (batch_ids, batch_zids, x, y, z, beta) in enumerate(loader):
                global_step += 1
                if global_step == total_sgd_steps:
                    return

                x, y, batch_ids = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), batch_ids.to(cu.get_device(), dtype=torch.int64)
                self._optimizer.zero_grad()
                
                # For xent loss, we need only pass unnormalized logits. https://stackoverflow.com/questions/49390842/cross-entropy-in-pytorch
                cls_out = self._model.forward(x)
                loss = self._xecri_perex(cls_out, y)
                loss = torch.dot(loss, trn_wts[batch_ids]) / torch.sum(trn_wts[batch_ids])
                
                loss.backward()
                self._optimizer.step()

                tq.set_postfix({"Loss": loss.item()})
                tq.update(1)

                if self._sw is not None:
                    self._sw.add_scalar("Loss", loss.item(), global_step)
            
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
        
            epoch_acc = self.accuracy()
            print(f"Epoch accuracy: {epoch_acc}")
            if self._sw is not None:
                self._sw.add_scalar("Epoch_Acc", epoch_acc, epoch)


    def grp_accuracy(self, X_test=None, y_test=None, Beta_test=None, *args, **kwargs) -> dict:
        """Adding some more functionality to our Shapenet dataset
        """

        assert X_test is None, "For now i assume that we will have to get grp_acc metrics only for test data"
        
        res_dict = super().grp_accuracy(X_test, y_test, Beta_test, *args, **kwargs)

        loader = self._tst_loader
        ideal_beta = self._tst_data._ideal_betas
        ideal_idxs = np.where(ideal_beta == 1)[0]
        non_ideal_idxs = np.where(ideal_beta == 0)[0]
        
        res_dict["ideal_accuracy"] = self.accuracy(loader=tu.get_loader_subset(loader, ideal_idxs))
        res_dict["non-ideal_accuracy"] = self.accuracy(loader=tu.get_loader_subset(loader, non_ideal_idxs))

        return res_dict
        
    def beta_accuracy(self, X_test=None, y_test=None, Beta_test=None, *args, **kwargs) -> dict:

        """Computes the accuracy on a per group basis

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]
            Beta_test ([type]): [description]

        Returns:
            dict: [description]
        """
        res_dict = super().beta_accuracy(X_test, y_test, Beta_test, *args, **kwargs)
        return res_dict


    @property
    def _def_name(self):
        return "resnet"
