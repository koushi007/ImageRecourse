from abc import ABC, abstractmethod, abstractproperty
import numpy as np
import abc
import utils.common_utils as cu
import data.syn_dataset as synds
import torch
import torch.utils.data as data_utils

class Data(ABC):
    """This is an abstract class for Dataset
    For us dataset is a tuple (x, y, z, beta, rho)
    For rho, if we regress logits, we have probabilities
    If we decide to regress loss, then we have losses
    """
    def __init__(self, X, y, Z, Beta, B_per_i=None) -> None:
        super().__init__()
        self.data_ids = np.arange(len(X))
        self.X = X
        self.y = y
        self.Z = Z
        self.Beta = Beta
        self.B_per_i = B_per_i
        self.classifier = None 
        self.Rho = None
        self.num_classes = len(set(y))
        self.classes = set(y)

    @property
    def _data_ids(self) -> np.array:
        return self.data_ids

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
    def _0INDy(self):
        """Returns -1, +1 y in {0, 1} format
        """
        if self._num_classes != 2:
            raise ValueError("This is not a binary dataset")
        if -1 not in set(self._y):
            raise ValueError("Use this API only for converting {-1, +1} -> {0, 1}")
        y = self._y.copy()
        y[y == -1] = 0
        return y
    
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
    
    # some useful functions
    @property
    def _num_data(self):
        return self._X.shape[0]
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

    def init_loader(self, ids, X, y, Z, Beta, shuffle=True, batch_size=None):
        T = torch.Tensor
        dataset = data_utils.TensorDataset(T(ids), T(X), T(y), T(Z), T(Beta))
        return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    def init_grp_loader(self, ids, X, y, Z, Beta, shuffle=True, batch_size=None):
        grp_arr = lambda arr : np.array(np.split(arr, int(len(arr) / self.B_per_i)))
        return self.init_loader(grp_arr(ids), grp_arr(X), grp_arr(y), grp_arr(Z), grp_arr(Beta), 
                                shuffle=shuffle, batch_size=int(batch_size / self._B_per_i))

    @abc.abstractmethod
    def get_loader(self, shuffle, bsz):
        raise NotImplementedError()

    @abc.abstractmethod
    def get_grp_loader(self, shuffle, bsz):
        raise NotImplementedError()



class SyntheticData(Data):
    def __init__(self, X, y, Z, Beta, B_per_i) -> None:
        super(SyntheticData, self).__init__(X, y, Z, Beta, B_per_i)
    
    def apply_recourse(self, data_id, betas):
        """Applies recourse to the specified data is and returns the recoursed x
        Args:
            data_id ([type]): [description]
            betas ([type]): [description]
        Returns:
            [type]: [description]
        """
        x_rec = []
        z = self._Z[data_id]
        return np.multiply(z, betas)
    
    def get_loader(self, shuffle, bsz):
        return self.init_loader(self._data_ids, self._X, self._0INDy, self._Z, self._Beta, shuffle=shuffle, batch_size=bsz)
    
    def get_grp_loader(self, shuffle, bsz):
        return self.init_grp_loader(self._data_ids, self._X, self._0INDy, self._Z, self._Beta, shuffle=shuffle, batch_size=bsz)

class DataHelper(ABC):
    def __init__(self) -> None:
        super().__init__()
        self.train = None
        self.test = None
    
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

    


class SyntheticDataHelper(DataHelper):
    def __init__(self, dim=10, prior=0.5, 
                    num_train=200, B_per_i=5, num_test=100, beta_noise=0.7, 
                    mean_start=0.5, mean_step=0.1, 
                    var_start=0.05, var_step=0.05) -> None:
        super().__init__()
        
        self.dim = dim
        self.prior = prior
        self.num_train = num_train 
        self.B_per_i = B_per_i # number of betas per z
        self.num_test = num_test

        # This is beta noise that tells us how many entries should be OFF
        self.beta_noise = beta_noise

        # mean spec
        self.mean_start = mean_start
        self.mean_step = mean_step
        # variance spec
        self.var_start = var_start
        self.var_step = var_step

        self.mean = {
            "pos": [mean_start-(idx*mean_step) for idx in range(10)],
            "neg": [-mean_start+(idx*mean_step) for idx in range(10)]
        }
        self.var = {
            "pos": [var_start+(idx*var_step) for idx in range(10)],
            "neg": [var_start+(idx*var_step) for idx in range(10)]
        }

        cu.set_seed()
        D_train, D_test = synds.sample_dataset(prior=self.prior,
                                                B_per_i = self.B_per_i,
                                                beta_noise=self.beta_noise,
                                                mean=self.mean, var=self.var,
                                                dim=self.dim,
                                                num_train=self.num_train, num_test=self.num_test)
        
        def process_data(dataset, B_per_i = None) -> SyntheticData:
            data = SyntheticData(dataset["x"], dataset["y"],
                                    dataset["z"], dataset["beta"], B_per_i=B_per_i)
            return data

        self._train, self._test = process_data(D_train, self.B_per_i), process_data(D_test)


    def __str__(self) -> str:
        return f"Bperi={self.B_per_i};bNois={self.beta_noise};mn={self.mean_start}-{self.mean_step};var={self.var_start}-{self.var_step}"


# %% Test cases
if __name__ == "__main__":
    sdh = SyntheticDataHelper()
    train, test = sdh.train, sdh.test
    assert train._X.shape == train._Z.shape, "wrong 1"
    assert train._X.shape == train._Beta.shape, "wrong 2"
    assert train._X[np.arange(100)].shape[0] == 100, "wrong 3"
    pass