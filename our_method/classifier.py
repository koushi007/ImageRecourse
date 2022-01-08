from abc import ABC
import torch.nn as nn
import torch
from copy import deepcopy
import torch.optim as optim
import our_method.data_helper as ourdh

class ModelHelper(ABC):
    def __init__(self, model:nn.Module, *args, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.__model_copy = None
        self.lr = 1e-3

        self.__init_kwargs(kwargs)
    
    def __init_kwargs(self, kwargs:dict):
        if "lr" in kwargs.keys():
            self.lr = kwargs["lr"]

    @property
    def _model(self):
        return self.model
    @_model.setter
    def _model(self, value):
        self.model = value

    def copy_model(self):
        """Stores a copy of the model
        """
        self.__model_copy = deepcopy(self._model.state_dict())
    
    def apply_copied_model(self):
        """Loads the weights of deep copied model to the origibal model
        """
        assert self.__model_copy != None
        self.model.load_state_dict(self.__model_copy)

    def clear_copy(self):
        """[summary]
        """
        self.__model_copy = None

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
    def _msecri(self):
        return nn.MSELoss()

    @property
    def _msecri_perex(self):
        return nn.MSELoss(reduction="none")
    

class LRModel(nn.Module):
    def __init__(self, in_dim, n_classes, *args, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.n_classes = n_classes

        self.in_layer = nn.Linear(in_dim, out_features=n_classes)
        self.sm = nn.Softmax()

    def forward_prob(self, input):
        out = self.in_layer(input)
        return self.sm(out)
    
    def forward(self, input):
        return self.in_layer(input)

class LRHelper(ModelHelper):  
    def __init__(self, in_dim, n_classes, *args, **kwargs) -> None:
        model = LRModel(in_dim=in_dim, n_classes=n_classes, args=args, kwargs=kwargs)
        self.in_dim = in_dim
        self.n_classes = n_classes

        super(LRHelper, self).__init__(model, args, kwargs)

        self.optimizer = optim.SGD(self._model.parameters(), lr=self._lr)

    def fit_data(self, x, y, epochs, steps, *args, **kwargs):
        pass
