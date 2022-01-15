import abc
from abc import ABC, abstractmethod, abstractproperty
import our_method.constants as constants
import numpy as np
import torch
import torch.utils.data as data_utils
import utils.common_utils as cu
import utils.torch_utils as tu


class Data(ABC):
    """This is an abstract class for Dataset
    For us dataset is a tuple (x, y, z, beta, rho)
    For rho, if we regress logits, we have probabilities
    If we decide to regress loss, then we have losses
    """
    def __init__(self, X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs) -> None:
        super().__init__()
        self.X = X
        self.y = y
        self.Z = Z

        assert len(Z) == len(y), "If the length of Z and y is not the same, we may have to repeat_interleave somewhere to make the rest of code easy to access."

        self.Beta = Beta
        self.B_per_i = B_per_i
        self.Siblings = Siblings
        self.Z_ids = Z_ids
        self.num_classes = len(set(y))
        self.classes = set(y)
        self.ideal_betas = ideal_betas

        self.data_ids = np.arange(len(X))
        self.num_Z = len(set(self.Z_ids))

        self.transform = None
        self.__init_kwargs(kwargs)

    def __init_kwargs(self, kwargs):
         if constants.TRANSFORM in kwargs:
            self.transform = kwargs[constants.TRANSFORM]

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
    def _Siblings(self):
        if self.Siblings is None:
            raise ValueError("Why are u calling siblings on the test/val data?")
        return self.Siblings
    
    @property
    def _ideal_betas(self):
        return self.ideal_betas
    @_ideal_betas.setter
    def _ideal_betas(self, value):
        raise ValueError("Pass it once in constructor. Why are u settig it again?")

# %% some useful functions
    
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
        return self._Siblings[data_ids]

    def get_loader(self, shuffle, batch_size):
        loader_args = {}
        if self.transform is not None:
            loader_args[constants.TRANSFORM] = self.transform
        return tu.init_loader(self._data_ids, self._Z_ids, self._X, self._y, self._Z, self._Beta, shuffle=shuffle, batch_size=batch_size, **loader_args)

    def get_grp_loader(self, shuffle, batch_size):
        loader_args = {}
        if self.transform is not None:
            loader_args[constants.TRANSFORM] = self.transform
        return tu.init_grp_loader(self._data_ids, self._Z_ids, self._X, self._y, self._Z, self._Beta, self._B_per_i,
                                     shuffle=shuffle, batch_size=batch_size, **loader_args)
    
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

class SyntheticData(Data):
    def __init__(self, X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs) -> None:
        super(SyntheticData, self).__init__(X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs)
    
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

class ShapenetData(Data):
    def __init__(self, X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs) -> None:
        super(ShapenetData, self).__init__(X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs)
    
    def apply_recourse(self, data_ids, betas:np.array):
        """Applies recourse to the specified data is and returns the recoursed x
        Args:
            data_ids ([type]): [description]
            betas ([type]): [description]
        Returns:
            [type]: [description]
        """
        raise NotImplementedError()
        



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

class ShapenetDataHelper(DataHelper):
    def __init__(self, train, test, val) -> None:
        super(ShapenetDataHelper, self).__init__(train, test, val)
