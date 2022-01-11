from abc import ABC, abstractmethod, abstractproperty
from os import XATTR_CREATE
from pathlib import Path
from socket import SocketIO
import warnings

from torch.utils.tensorboard.writer import SummaryWriter
from our_method.data_helper import Data

import our_method.nn_phi as ourphi
import our_method.nn_psi as ourpsi
import our_method.nn_theta as ourth
import our_method.data_helper as ourd
import torch
import torch.utils.data as data_utils
from our_method.recourse import RecourseHelper
import utils.common_utils as cu
import numpy as np
import torch.nn as nn
import utils.torch_utils as tu
from tqdm import tqdm
import torch.optim as optim

class MethodsHelper(ABC):
    """
    This is a base class for all models that provide recourse to our Synthetic dataset.
    Donot instantiate objects out of tjis class.
    """
    def __init__(self, dh:ourd.DataHelper, nnth:ourth.NNthHelper, nnphi:ourphi.NNPhiHelper, 
                        nnpsi:ourpsi.NNPsiHelper, rechlpr:RecourseHelper, *args, **kwargs) -> None:
        
        self.dh = dh
        self.nnth = nnth
        self.nnphi = nnphi
        self.nnpsi = nnpsi
        self.rech = rechlpr

        self.R = torch.Tensor(self.rech._R).to(cu.get_device(), dtype=torch.int64)
        self.rech.set_Sij()
        self.Sij = torch.Tensor(self.rech._Sij).to(cu.get_device(), dtype=torch.int64)

        self.optim = None
        self.phi_optim = None
        self.th_optim = None
        self.psi_optim = None

        self.lr = 1e-3
        self.sw = None
        self.batch_size = 5 * self.dh._train._B_per_i
        self.pretrn_models = {
            "th": False,
            "phi": False,
            "psi": False
        }
        self.__init_kwargs(kwargs)
        self.__init_loader()


# %% inits
    def __init_kwargs(self, kwargs:dict):
        if "lr" in kwargs.keys():
            self.lr = kwargs["lr"]
        if "summarywriter" in kwargs.keys():
            self.sw = kwargs["summarywriter"]
        if "batch_size" in kwargs.keys():
            self.batch_size = kwargs["batch_size"]
        if "pretrn_th_phi_psi" in kwargs.keys():
            assert isinstance(kwargs["pretrn_th_phi_psi"], dict), "Please pass a dictionary to me"
            self.pretrn_models = kwargs["pretrn_th_phi_psi"]
            if not self.pretrn_models["th"]:
                tu.init_weights(self._thmodel)
            if not self.pretrn_models["phi"]:
                tu.init_weights(self._phimodel)
            if not self.pretrn_models["psi"]:
                tu.init_weights(self._psimodel)

    def __init_loader(self):
        """We uill use the dh loaders only here
        """
        # For this we can simply use the loaders available in nnth
        pass

# %% Properties
    @property
    def _thmodel(self):
        return self.nnth._model
    @_thmodel.setter
    def _thmodel(self, state_dict):
        self.nnth._model.load_state_dict(state_dict) 

    @property
    def _phimodel(self):
        return self.nnphi._phimodel
    @_phimodel.setter
    def _phimodel(self, state_dict):
        self.nnphi._phimodel.load_state_dict(state_dict) 

    @property
    def _psimodel(self):
        return self.nnpsi._psimodel
    @_psimodel.setter
    def _psimodel(self, state_dict):
        self.nnpsi._psimodel.load_state_dict(state_dict) 

    @property
    def _nnth(self) -> ourth.NNthHelper:
        return self.nnth
    
    @property
    def _nnphi(self) -> ourphi.NNPhiHelper:
        return self.nnphi

    @property
    def _nnpsi(self) -> ourpsi.NNPsiHelper:
        return self.nnpsi

    @property
    def _R(self):
        return self.R

    @property
    def _Sij(self):
        return self.Sij

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
        return self.nnth._trn_loader
    @property
    def _trngrp_loader(self):
        return self.nnth._trn_data.get_grp_loader(shuffle=True, batch_size=self.batch_size)

    @property
    def _tst_loader(self):
        return self.nnth._tst_loader

    @property
    def _val_loader(self):
        return self.nnth._val_loader

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
    def _Xdim(self):
        return self._trn_data._X.shape[1]
    @property
    def _Betadim(self):
        return self._trn_data._Beta.shape[1]
    @property
    def _num_classes(self):
        return self._trn_data._num_classes

    @property
    def _optim(self) -> optim.Optimizer:
        return self.optim
    @_optim.setter
    def _optim(self, value):
        self.optim = value

    @property
    def _thoptim(self) -> optim.Optimizer:
        return self.th_optim
    @_thoptim.setter
    def _thoptim(self, value):
        self.th_optim = value

    @property
    def _phioptim(self) -> optim.Optimizer:
        return self.phi_optim
    @_phioptim.setter
    def _phioptim(self, value):
        self.phi_optim = value

    @property
    def _psioptim(self) -> optim.Optimizer:
        return self.psi_optim
    @_psioptim.setter
    def _psioptim(self, value):
        self.psi_optim = value

    @property
    def _lr(self) -> float:
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

    @property
    def _KLCriterion_rednone(self):
        return nn.KLDivLoss(reduction="none")

    @property
    def _sel_nzro(self):
        sel_nonzero = lambda t, sij : torch.squeeze(t[torch.nonzero(sij)])
        return sel_nonzero
    
    @property
    def _sel_zro(self):
        sel_zero = lambda t, sij : torch.squeeze(1-t[torch.nonzero(sij)])
        return sel_zero


# %% abstract methods

    @abstractmethod
    def fit_epoch(self, epoch, loader=None, *args, **kwargs):
        raise NotImplementedError()
    
    @abstractproperty
    def _def_name(self):
        pretrn_sfx = "pretrn="
        if self.pretrn_models["th"]:
            pretrn_sfx += "Th"
        if self.pretrn_models["phi"]:
            pretrn_sfx += "Phi"
        if self.pretrn_models["psi"]:
            pretrn_sfx += "Psi"
        return pretrn_sfx + "-"

# %% some utilities

    def entropy_bias(self, loader=None, epochs=5):
        """This bias training is ap[plicable only to the NN_phi model to make it have an uniform 
        prior over all betas

        Args:
            loader ([type], optional): [description]. Defaults to None.
            epochs (int, optional): [description]. Defaults to 5.
        """
        self._phimodel.train()
        if loader is None:
            loader = self._trn_loader

        step = 0
        target = torch.tensor(torch.ones(self._batch_size, self._Xdim)).to(cu.get_device())
        target = target / self._Xdim

        for epoch in range(epochs):
            tq = tqdm(total=len(loader), desc="Loss")
            for data_ids, Z_ids, X, _, _, Beta in loader:
                step += 1
                X, Beta = X.to(cu.get_device()), Beta.to(cu.get_device())
                self._phioptim.zero_grad()

                y_preds = self._phimodel(X, Beta)
                y_preds = torch.softmax(y_preds, dim=1)
                y_preds = torch.log(y_preds)

                loss = self._KLCriterion_rednone(y_preds, target)
                loss.backward()
                self._phioptim.step()
                # self._sw.add_scalar("Ent-Prior", loss.item(), step)
                tq.set_description(f"Loss: {loss.item()}")
                tq.update(1)

    def accuracy(self, X_test = None, y_test=None, *args, **kwargs) -> float:
        if X_test is None:
            X_test = self._tst_data._X
            y_test = self._tst_data._y
        return self._nnth.accuracy(X_test, y_test, *args, **kwargs)

    def grp_accuracy(self, X_test: np.array = None, Beta_test: np.array = None, y_test: np.arange=None, *args, **kwargs) -> dict:
        if X_test is None:
            X_test = self._tst_data._X
            y_test = self._tst_data._y
            Beta_test = self._tst_data._Beta
        res_dict = {}
        for beta_id in range(self._Betadim):
            beta_samples = np.where(Beta_test[:, beta_id] == 1)
            res_dict[beta_id] = self.accuracy(X_test[beta_samples], y_test[beta_samples])
        return res_dict

    
    def predict_labels(self, X, *args, **kwargs) -> np.array:
        return self._nnth.predict_labels(X)

    def predict_proba(self, X, *args, **kwargs) -> np.array:
        return self._nnth.predict_proba(XATTR_CREATE)

    def predict_betas(self, X, Beta) -> np.array:
        return self._nnphi.predict_beta(X, Beta)

    def predict_r(self, X, Beta, *artgs, **kwargs) -> np.array:
        return self._nnpsi.predict_r(X, Beta)
    
    def save_model_defname(self, suffix = ""):
        state_dict = {
            "phi": self._phimodel.state_dict(),
            "psi": self._psimodel.state_dict(),
            "th": self._thmodel.state_dict()
        }
        dir = self._def_dir
        dir.mkdir(exist_ok=True, parents=True)
        torch.save(state_dict, self._def_dir / (self._def_name + suffix + ".pt"))
    
    def load_model_defname(self, suffix=""):
        state_dict = torch.load(self._def_dir / (self._def_name + suffix + ".pt"), map_location=cu.get_device())
        self._thmodel.load_state_dict(state_dict["th"])
        self._phimodel.load_state_dict(state_dict["phi"])
        self._psimodel.load_state_dict(state_dict["psi"])
        print(f"Models loader from {str(self._def_dir / (self._def_name + suffix + '.pt'))}")


class BaselineHelper(MethodsHelper):
    def __init__(self, dh: ourd.DataHelper, nnth: ourth.NNthHelper, nnphi: ourphi.NNPhiHelper, nnpsi: ourpsi.NNPsiHelper, rechlpr: RecourseHelper, *args, **kwargs) -> None:
        super().__init__(dh, nnth, nnphi, nnpsi, rechlpr, *args, **kwargs)

        # Set all the optimizers
        self._optim = optim.AdamW([
            {'params': self._phimodel.parameters()},
            {'params': self._thmodel.parameters()},
            {'params': self._psimodel.parameters()}
        ], lr=self._lr)

        self._thoptim = optim.AdamW([
            {'params': self._thmodel.parameters()}
        ], lr = self._lr)

        self._phioptim = optim.AdamW([
            {'params': self._phimodel.parameters()}
        ], lr = self._lr)

        self._psioptim = optim.AdamW([
            {'params': self._psimodel.parameters()}
        ], lr = self._lr)

    def _def_name(self):
        return super()._def_name + "baseline"


    def fit_epoch(self, epoch, loader=None, *args, **kwargs):
        
        inter_iters = -1
        if "interleave_iters" in kwargs:
            inter_iters = kwargs["interleave_iters"]

        self._thmodel.train()
        self._phimodel.train()
        self._psimodel.train()

        if loader is None:
            loader = self._trngrp_loader
        else:
            warnings.warn("Are u sure that u are passing the right group loader?")
        
        global_step = epoch * len(loader)
        tq = tqdm(total=len(loader), desc="Loss")

        num_grp = self._trn_data.B_per_i
        
        for local_step, (dataid_grp, Zid_grp, X_grp, y_grp, Z_grp, Beta_grp) in enumerate(loader):
            util_grp = 0
            for dataid, zid, x, y, z, beta in zip(dataid_grp, Zid_grp, X_grp, y_grp, Z_grp, Beta_grp):
                
                x, y, beta = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), beta.to(cu.get_device())
                self._optim.zero_grad()
                
                # this is P(\beta_ir | ij)
                rec_beta = self._phimodel.forward(x, beta)
                rec_beta = torch.sigmoid(rec_beta)

                rec_beta_agg = rec_beta.repeat_interleave(num_grp, dim=0)
                beta_agg = beta.repeat(num_grp, 1)
                rec_beta_loss = torch.mul(beta_agg, torch.log(rec_beta_agg+1e-5)) + torch.mul(1-beta_agg, torch.log(1-rec_beta_agg+1e-5))
                rec_beta_loss = torch.sum(rec_beta_loss, dim=1)

                cls_out = self._thmodel.forward(x)
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
            self._optim.step()
            self._sw.add_scalar("Loss", loss.item(), global_step+local_step)
            tq.set_description(f"Loss: {loss.item()}")
            tq.update(1)

class BaselineKLHelper(MethodsHelper):
    def __init__(self, dh: ourd.DataHelper, nnth: ourth.NNthHelper, nnphi: ourphi.NNPhiHelper, nnpsi: ourpsi.NNPsiHelper, rechlpr: RecourseHelper, *args, **kwargs) -> None:
        super().__init__(dh, nnth, nnphi, nnpsi, rechlpr, *args, **kwargs)


        # Set all the optimizers
        self._optim = optim.AdamW([
            {'params': self._phimodel.parameters()},
            {'params': self._thmodel.parameters()},
            {'params': self._psimodel.parameters()}
        ], lr=self._lr)


        self._thoptim = optim.AdamW([
            {'params': self._thmodel.parameters()}
        ], lr = self._lr)

        self._phioptim = optim.AdamW([
            {'params': self._phimodel.parameters()}
        ], lr = self._lr)

        self._psioptim = optim.AdamW([
            {'params': self._psimodel.parameters()}
        ], lr = self._lr)

    def _def_name(self):
        return super()._def_name + "baseline"


    def fit_epoch(self, epoch, loader=None, *args, **kwargs):
        
        inter_iters = -1
        if "interleave_iters" in kwargs:
            inter_iters = kwargs["interleave_iters"]

        self._thmodel.train()
        self._phimodel.train()
        self._psimodel.train()

        if loader is None:
            loader = self._trngrp_loader
        else:
            warnings.warn("Are u sure that u are passing the right group loader?")
        
        global_step = epoch * len(loader)
        tq = tqdm(total=len(loader), desc="Loss")

        num_grp = self._trn_data.B_per_i
        uni_kl_targets = torch.ones(num_grp*num_grp, self._Betadim) / self._Betadim
        uni_kl_targets = uni_kl_targets.to(cu.get_device())
        ent_decay = torch.tensor([0.9]).to(cu.get_device())
        
        for local_step, (dataid_grp, Zid_grp, X_grp, y_grp, Z_grp, Beta_grp) in enumerate(loader):
            util_grp = 0
            for dataid, zid, x, y, z, beta in zip(dataid_grp, Zid_grp, X_grp, y_grp, Z_grp, Beta_grp):
                
                x, y, beta = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), beta.to(cu.get_device())
                self._optim.zero_grad()
                
                # this is P(\beta_ir | ij)
                rec_beta = self._phimodel.forward(x, beta)
                rec_beta = torch.softmax(rec_beta, dim=1)
                rec_beta = torch.log(rec_beta + 1e-5)

                rec_beta_agg = rec_beta.repeat_interleave(num_grp, dim=0)
                beta_agg = beta.repeat(num_grp, 1)

                rec_beta_loss_ir = -self._KLCriterion_rednone(rec_beta_agg, beta_agg / torch.sum(beta_agg, dim=1).view(-1, 1))
                rec_beta_loss_uni = self._KLCriterion_rednone(rec_beta_agg, uni_kl_targets)

                rec_beta_loss_ir = torch.sum(rec_beta_loss_ir, dim=1)
                rec_beta_loss_uni = torch.sum(rec_beta_loss_uni, dim=1)
                self._sw.add_scalar("-KL(betair || beta|xij)", torch.mean(rec_beta_loss_ir).item(), global_step+local_step)
                self._sw.add_scalar("KL(betair|| uni)", torch.mean(rec_beta_loss_uni).item(), global_step+local_step)
                rec_beta_loss = rec_beta_loss_ir + torch.pow(ent_decay, epoch+1) * rec_beta_loss_uni

                cls_out = self._thmodel.forward(x)
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
            self._optim.step()
            self._sw.add_scalar("Loss", loss.item(), global_step+local_step)
            tq.set_description(f"Loss: {loss.item()}")
            tq.update(1)

class Method1Helper(MethodsHelper):
    def __init__(self, dh: ourd.DataHelper, nnth: ourth.NNthHelper, nnphi: ourphi.NNPhiHelper, nnpsi: ourpsi.NNPsiHelper, rechlpr: RecourseHelper, *args, **kwargs) -> None:
        super().__init__(dh, nnth, nnphi, nnpsi, rechlpr, *args, **kwargs)


        # Set all the optimizers
        self._optim = optim.AdamW([
            {'params': self._phimodel.parameters()},
            {'params': self._thmodel.parameters()},
            {'params': self._psimodel.parameters()}
        ], lr=self._lr)

        self._thoptim = optim.AdamW([
            {'params': self._thmodel.parameters()}
        ], lr = self._lr)

        self._phioptim = optim.AdamW([
            {'params': self._phimodel.parameters()}
        ], lr = self._lr)

        self._psioptim = optim.AdamW([
            {'params': self._psimodel.parameters()}
        ], lr = self._lr)

    def _def_name(self):
        return super()._def_name + "baselinekl"

    def fit_epoch(self, epoch, loader=None, *args, **kwargs):
        
        inter_iters = -1
        if "interleave_iters" in kwargs:
            inter_iters = kwargs["interleave_iters"]

        self._phimodel.train()
        self._thmodel.train()

        if loader is None:
            loader = self._trngrp_loader
        
        global_step = epoch * len(loader)
        tq = tqdm(total=len(loader), desc="Loss")

        num_grp = self.dh._train.B_per_i
        uni_kl_targets = torch.ones(num_grp*num_grp, self._Betadim) / self._Betadim
        uni_kl_targets = uni_kl_targets.to(cu.get_device())
        ent_decay = torch.tensor([0.9]).to(cu.get_device())
    
        for local_step, (dataid_grp, Zid_grp, X_grp, y_grp, Z_grp, Beta_grp) in enumerate(loader):
            util_grp = 0
            for dataids, zid, x, y, z, beta in zip(dataid_grp, Zid_grp, X_grp, y_grp, Z_grp, Beta_grp):
                
                dataids, x, y, beta = dataids.to(cu.get_device()), x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), beta.to(cu.get_device())
                self._optim.zero_grad()


                rec = torch.Tensor([entry in self._R for entry in dataids])
                Sij = self._Sij[dataids]
                rec_Sij = self._sel_nzro(Sij, rec)

                # this is P(\beta_ir | ij)
                rec_beta = self._phimodel.forward(x, beta)
                rec_beta = torch.softmax(rec_beta, dim=1)
                rec_beta = torch.log(rec_beta + 1e-5)

                rec_beta = self._sel_nzro(rec_beta, rec)
                rec_beta_agg = rec_beta.repeat_interleave(num_grp, dim=0)
                
                beta_agg = self._sel_nzro(beta, rec).repeat(num_grp, 1)

                rec_beta_loss_ir = -self._KLCriterion_rednone(reduction="none")(rec_beta_agg, beta_agg / torch.sum(beta_agg, dim=1).view(-1, 1))
                rec_beta_loss_uni = self._KLCriterion_rednone(reduction="none")(rec_beta_agg, uni_kl_targets[:rec_beta_agg.size(0)])

                rec_beta_loss_ir = torch.sum(rec_beta_loss_ir, dim=1)
                rec_beta_loss_uni = torch.sum(rec_beta_loss_uni, dim=1)
                self._sw.add_scalar("-KL(betair || beta|xij)", torch.mean(rec_beta_loss_ir).item(), global_step+local_step)
                self._sw.add_scalar("KL(betair|| uni)", torch.mean(rec_beta_loss_uni).item(), global_step+local_step)
                rec_beta_loss = rec_beta_loss_ir + torch.pow(ent_decay, epoch+1) * rec_beta_loss_uni

                cls_out = self._thmodel.forward(x)
                cls_out = torch.softmax(cls_out, dim=1)
                cls_out = torch.gather(cls_out, 1, y.view(-1, 1)).squeeze()
                cls_out = torch.log(cls_out + 1e-5)
                cls_loss = cls_out.repeat(num_grp)

                
                
                nr_cls_loss = self._sel_zro(cls_out, rec)
                



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
            self._optim.step()
            self._sw.add_scalar("Loss", loss.item(), global_step+local_step)
            tq.set_description(f"Loss: {loss.item()}")
            tq.update(1)

# class Method2Helper(MethodsHelper):
#     def __init__(self, dh: ourd.DataHelper, nnth: ourth.NNthHelper, nnphi: ourphi.NNPhiHelper, nnpsi: ourpsi.NNPsiHelper, rechlpr: RecourseHelper, *args, **kwargs) -> None:
#         super().__init__(dh, nnth, nnphi, nnpsi, rechlpr, *args, **kwargs)


        # # Set all the optimizers
        # self._optim = optim.AdamW([
        #     {'params': self._phimodel.parameters()},
        #     {'params': self._thmodel.parameters()},
        #     {'params': self._psimodel.parameters()}
        # ], lr=self._lr)


#         self._thoptim = optim.AdamW([
#             {'params': self._thmodel.parameters()}
#         ], lr = self._lr)

#         self._phioptim = optim.AdamW([
#             {'params': self._phimodel.parameters()}
#         ], lr = self._lr)

#         self._psioptim = optim.AdamW([
#             {'params': self._psimodel.parameters()}
#         ], lr = self._lr)

#     def _def_name(self):
#         return super()._def_name + "method1"

#     def fit_epoch(self, epoch, loader=None, *args, **kwargs):
        
#         inter_iters = -1
#         if "interleave_iters" in kwargs:
#             inter_iters = kwargs["interleave_iters"]

#         self._phimodel.train()
#         self._thmodel.train()

#         if loader is None:
#             loader = self._trngrp_loader
        
#         global_step = epoch * len(loader)
#         tq = tqdm(total=len(loader), desc="Loss")

#         num_grp = self.dh._train.B_per_i
#         uni_kl_targets = torch.ones(num_grp*num_grp, self._Betadim) / self._Betadim
#         uni_kl_targets = uni_kl_targets.to(cu.get_device())
#         ent_decay = torch.tensor([0.9]).to(cu.get_device())
    
#         for local_step, (dataid_grp, Zid_grp, X_grp, y_grp, Z_grp, Beta_grp) in enumerate(loader):
#             util_grp = 0
#             for dataid, zid, x, y, z, beta in zip(dataid_grp, Zid_grp, X_grp, y_grp, Z_grp, Beta_grp):
                
#                 x, y, beta = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), beta.to(cu.get_device())
#                 self._optim.zero_grad()
                
#                 # this is P(\beta_ir | ij)
#                 rec_beta = self._phimodel.forward(x, beta)
#                 rec_beta = torch.softmax(rec_beta, dim=1)
#                 rec_beta = torch.log(rec_beta + 1e-5)

#                 rec_beta_agg = rec_beta.repeat_interleave(num_grp, dim=0)
#                 beta_agg = beta.repeat(num_grp, 1)

#                 rec_beta_loss_ir = -self._KLCriterion(rec_beta_agg, beta_agg / torch.sum(beta_agg, dim=1).view(-1, 1))
#                 rec_beta_loss_uni = self._KLCriterion(rec_beta_agg, uni_kl_targets)


#                 rec_beta_loss_ir = torch.sum(rec_beta_loss_ir, dim=1)
#                 rec_beta_loss_uni = torch.sum(rec_beta_loss_uni, dim=1)
#                 self._sw.add_scalar("-KL(betair || beta|xij)", torch.mean(rec_beta_loss_ir).item(), global_step+local_step)
#                 self._sw.add_scalar("KL(betair|| uni)", torch.mean(rec_beta_loss_uni).item(), global_step+local_step)
#                 rec_beta_loss = rec_beta_loss_ir + torch.pow(ent_decay, epoch+1) * rec_beta_loss_uni

#                 cls_out = self._thmodel.forward(x)
#                 cls_out = torch.softmax(cls_out, dim=1)
#                 cls_out = torch.gather(cls_out, 1, y.view(-1, 1)).squeeze()
#                 cls_out = torch.log(cls_out + 1e-5)
#                 cls_loss = cls_out.repeat(num_grp)

#                 if inter_iters != -1:
#                     if (local_step % (2*inter_iters)) % inter_iters == 0:
#                         do_rec, do_cls = 0, 1
#                     else:
#                         do_rec, do_cls = 1, 0
#                 else:
#                     do_rec, do_cls = 1, 1
#                 util = (do_rec * rec_beta_loss) + (do_cls * cls_loss)

#                 util = util.view(num_grp, -1)
#                 util, max_idxs = torch.max(util, dim=0)
#                 util = torch.sum(util)
                
#                 util_grp += util

#                 cls_loss_sw = torch.gather(cls_loss.view(num_grp, -1), 0, max_idxs.view(-1, 1))
#                 rec_loss_sw = torch.gather(rec_beta_loss.view(num_grp, -1), 0, max_idxs.view(-1, 1))
#                 self._sw.add_scalar("cls_loss", torch.mean(cls_loss_sw), global_step+local_step)
#                 self._sw.add_scalar("rec_loss", torch.mean(rec_loss_sw), global_step+local_step)


#             loss = -util_grp/len(X_grp)
#             loss.backward()
#             self._optimizer.step()
#             self._sw.add_scalar("Loss", loss.item(), global_step+local_step)
#             tq.set_description(f"Loss: {loss.item()}")
#             tq.update(1)

# class MethodRwdHelper(MethodsHelper):
#     def __init__(self, dh: ourd.DataHelper, nnth: ourth.NNthHelper, nnphi: ourphi.NNPhiHelper, nnpsi: ourpsi.NNPsiHelper, rechlpr: RecourseHelper, *args, **kwargs) -> None:
#         super().__init__(dh, nnth, nnphi, nnpsi, rechlpr, *args, **kwargs)

    #    # Set all the optimizers
    #     self._optim = optim.AdamW([
    #         {'params': self._phimodel.parameters()},
    #         {'params': self._thmodel.parameters()},
    #         {'params': self._psimodel.parameters()}
    #     ], lr=self._lr)

#         self._thoptim = optim.AdamW([
#             {'params': self._thmodel.parameters()}
#         ], lr = self._lr)

#         self._phioptim = optim.AdamW([
#             {'params': self._phimodel.parameters()}
#         ], lr = self._lr)

#         self._psioptim = optim.AdamW([
#             {'params': self._psimodel.parameters()}
#         ], lr = self._lr)

#     def _def_name(self):
#         return super()._def_name + "method1"

#     def fit_epoch(self, epoch, loader=None, *args, **kwargs):
        
#         inter_iters = -1
#         if "interleave_iters" in kwargs:
#             inter_iters = kwargs["interleave_iters"]

#         self._phimodel.train()
#         self._thmodel.train()

#         if loader is None:
#             loader = self._trngrp_loader
        
#         global_step = epoch * len(loader)
#         tq = tqdm(total=len(loader), desc="Loss")

#         num_grp = self.dh._train.B_per_i
#         uni_kl_targets = torch.ones(num_grp*num_grp, self._Betadim) / self._Betadim
#         uni_kl_targets = uni_kl_targets.to(cu.get_device())
#         ent_decay = torch.tensor([0.9]).to(cu.get_device())
    
#         for local_step, (dataid_grp, Zid_grp, X_grp, y_grp, Z_grp, Beta_grp) in enumerate(loader):
#             util_grp = 0
#             for dataid, zid, x, y, z, beta in zip(dataid_grp, Zid_grp, X_grp, y_grp, Z_grp, Beta_grp):
                
#                 x, y, beta = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), beta.to(cu.get_device())
#                 self._optim.zero_grad()
                
#                 # this is P(\beta_ir | ij)
#                 rec_beta = self._phimodel.forward(x, beta)
#                 rec_beta = torch.softmax(rec_beta, dim=1)
#                 rec_beta = torch.log(rec_beta + 1e-5)

#                 rec_beta_agg = rec_beta.repeat_interleave(num_grp, dim=0)
#                 beta_agg = beta.repeat(num_grp, 1)

#                 rec_beta_loss_ir = -self._KLCriterion(reduction="none")(rec_beta_agg, beta_agg / torch.sum(beta_agg, dim=1).view(-1, 1))
#                 rec_beta_loss_uni = self._KLCriterion(reduction="none")(rec_beta_agg, uni_kl_targets)


#                 rec_beta_loss_ir = torch.sum(rec_beta_loss_ir, dim=1)
#                 rec_beta_loss_uni = torch.sum(rec_beta_loss_uni, dim=1)
#                 self._sw.add_scalar("-KL(betair || beta|xij)", torch.mean(rec_beta_loss_ir).item(), global_step+local_step)
#                 self._sw.add_scalar("KL(betair|| uni)", torch.mean(rec_beta_loss_uni).item(), global_step+local_step)
#                 rec_beta_loss = rec_beta_loss_ir + torch.pow(ent_decay, epoch+1) * rec_beta_loss_uni

#                 cls_out = self._thmodel.forward(x)
#                 cls_out = torch.softmax(cls_out, dim=1)
#                 cls_out = torch.gather(cls_out, 1, y.view(-1, 1)).squeeze()
#                 cls_out = torch.log(cls_out + 1e-5)
#                 cls_loss = cls_out.repeat(num_grp)

#                 if inter_iters != -1:
#                     if (local_step % (2*inter_iters)) % inter_iters == 0:
#                         do_rec, do_cls = 0, 1
#                     else:
#                         do_rec, do_cls = 1, 0
#                 else:
#                     do_rec, do_cls = 1, 1
#                 util = (do_rec * rec_beta_loss) + (do_cls * cls_loss)

#                 util = util.view(num_grp, -1)
#                 util, max_idxs = torch.max(util, dim=0)
#                 util = torch.sum(util)
                
#                 util_grp += util

#                 cls_loss_sw = torch.gather(cls_loss.view(num_grp, -1), 0, max_idxs.view(-1, 1))
#                 rec_loss_sw = torch.gather(rec_beta_loss.view(num_grp, -1), 0, max_idxs.view(-1, 1))
#                 self._sw.add_scalar("cls_loss", torch.mean(cls_loss_sw), global_step+local_step)
#                 self._sw.add_scalar("rec_loss", torch.mean(rec_loss_sw), global_step+local_step)


#             loss = -util_grp/len(X_grp)
#             loss.backward()
#             self._optimizer.step()
#             self._sw.add_scalar("Loss", loss.item(), global_step+local_step)
#             tq.set_description(f"Loss: {loss.item()}")
#             tq.update(1)
