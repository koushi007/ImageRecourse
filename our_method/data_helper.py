from abc import ABC, abstractmethod, abstractproperty
from threading import local
from typing import List
import numpy as np
import abc

from torch._C import Value
import utils.common_utils as cu
import baseline.data.syn_dataset as synds
import torch
import torch.utils.data as data_utils

class Data(ABC):
    """This is an abstract class for Dataset
    For us dataset is a tuple (x, y, z, beta, rho)
    For rho, if we regress logits, we have probabilities
    If we decide to regress loss, then we have losses
    """
    def __init__(self, X, y, Z, Beta, B_per_i, siblings, Z_ids) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.Z = Z
        self.Beta = Beta
        self.B_per_i = B_per_i
        self.siblings = siblings
        self.Z_ids = Z_ids
        self.num_classes = len(set(y))
        self.classes = set(y)

        self.data_ids = np.arange(len(X))
        self.num_Z = len(set(self.Z_ids))

    @property
    def _data_ids(self) -> np.array:
        return self.data_ids

    @property
    def _Z_ids(self) -> np.array:
        return self.Z_ids

    @property
    def _X(self) -> np.array:
        return self.X
    @_X.setter
    def _X(self, value):
        self.X = value
    
    @property
    def _y(self) -> np.array:
       return self.y
    @_y.setter
    def _y(self, value):
        self.y = value
    
    @property
    def _Z(self) -> np.array:
        return self.Z
    @_Z.setter
    def _Z(self, value):
        self.Z = value
    
    @property
    def _Beta(self) -> np.array:
        return self.Beta
    @_Beta.setter
    def _Beta(self, value):
        self.Beta = value

    @property
    def _B_per_i(self):
        if self.B_per_i is None:
            raise ValueError("You are perhaps tryint to get this variable for test data which is illegal")
        return self.B_per_i

    @property
    def _siblings(self):
        if self.siblings is None:
            raise ValueError("Why are u calling siblings on the test/val data?")
        return self.siblings
    
    def get_instances(self, data_ids:np.array):
        """Returns ij data ids in order:
            x
            y
            z
            Beta

        Args:
            data_ids (np.array): [description]

        Returns:
            X, y, Z, Beta in order
        """
        if type(data_ids) == type([]):
            data_ids = np.array(data_ids)
        return self._X[data_ids], self._y[data_ids], self._Z[data_ids], self._Beta[data_ids]
    
    def get_Zinstances(self, zids:np.array):
        """Finds z id of all the ij instances given in the data_ids
        Then returns all the items in the Z group in order
            zids
            x
            y
            z
            beta

        Args:
            data_ids (np.array): [description]

        Returns:
            X, y, Z, Beta
        """
        if type(zids) == type([]):
            zids = np.array(zids)
        zids = [np.where(self._Z_ids == entry) for entry in zids]
        zids = np.array(zids).flatten()
        return zids, self._X[zids], self._y[zids], self._Z[zids], self._Beta[zids]
    
    def get_siblings_intances(self, data_ids):
        if type(data_ids) == type([]):
            data_ids = np.aray(data_ids)
        assert type(data_ids) == type(np.array([1])), "We expect even a scalar index to be passed as an array"
        return self._siblings[data_ids]
    

    # some useful functions
    @property
    def _num_data(self):
        return len(self.data_ids)

    @property
    def _num_Z(self):
        return self.num_Z

    @property
    def _Xdim(self):
        return self._X.shape[1]
    @property
    def _Betadim(self):
        return self._Beta.shape[1]
    @property
    def _num_classes(self):
        return self.num_classes
    @property
    def _classes(self):
        return self.classes

    @abc.abstractmethod
    def apply_recourse(self, data_id, betas):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_loader(self, shuffle, batch_size):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_grp_loader(self, shuffle, batch_size):
        raise NotImplementedError()

class SyntheticData(Data):
    def __init__(self, X, y, Z, Beta, B_per_i, siblings, Z_ids) -> None:
        super(SyntheticData, self).__init__(X, y, Z, Beta, B_per_i, siblings, Z_ids)
    
    def apply_recourse(self, data_ids, betas:np.array):
        """Applies recourse to the specified data is and returns the recoursed x
        Args:
            data_ids ([type]): [description]
            betas ([type]): [description]
        Returns:
            [type]: [description]
        """
        _, _, z, _ = self.get_instances(data_ids)
        assert z.shape() == betas.shape(), "Why the hell are the shapes inconsistent?"
        return np.multiply(z, betas)
    
    def get_loader(self, shuffle, bsz):
        return cu.init_loader(self._data_ids, self._Z_ids, self._X, self._y, self._Z, self._Beta, shuffle=shuffle, batch_size=bsz)

    def get_grp_loader(self, shuffle, bsz):
        return cu.init_grp_loader(self._data_ids, self._Z_ids, self._X, self._y, self._Z, self._Beta, self._B_per_i,
                                     shuffle=shuffle, batch_size=bsz)

class DataHelper(ABC):
    def __init__(self, train, test, val) -> None:
        super().__init__()
        self.train = train
        self.test = test
        self.val = val
    
    @property
    def _train(self) -> Data:
        return self.train
    @_train.setter
    def _train(self, value):
        self.train = value
    
    @property
    def _test(self) -> Data:
        return self.test
    @_test.setter
    def _test(self, value):
        self.test = value

    @property
    def _val(self) -> Data:
        return self.val
    @_val.setter
    def _val(self, value):
        self.val = value

class SyntheticDataHelper(DataHelper):
    def __init__(self, train, test, val) -> None:
        super(SyntheticDataHelper, self).__init__(train, test, val)

