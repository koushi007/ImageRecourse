from numpy.core.defchararray import index
import torch.nn as nn
import torch
from torchvision import models as tv_models

class LRModel(nn.Module):
    def __init__(self, in_dim, n_classes, *args, **kwargs):
        super().__init__()

        self.in_dim = in_dim
        self.n_classes = n_classes

        self.in_layer = nn.Linear(in_dim, out_features=n_classes)
        self.sm = nn.Softmax(dim=1)

    def forward_proba(self, input):
        out = self.in_layer(input)
        return self.sm(out)
    
    def forward(self, input):
        return self.in_layer(input)
    
    def forward_labels(self, input):
        probs = self.forward_proba(input)
        probs, labels = torch.max(probs, dim=1)
        return labels

class FNN(nn.Module):
    """creates a Feed Forward Neural network with the specified Architecture
        nn ([type]): [description]
    """
    def __init__(self, in_dim, out_dim, nn_arch, prefix, *args, **kwargs):
        """Creates a basic embedding block
        Args:
            in_dim ([type]): [description]
            embed_arch ([type]): [description]
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nn_arch = nn_arch
        self.prefix = prefix

        assert nn_arch[0] != in_dim and nn_arch[-1] != out_dim, "Assuming that we generally keep only bottleneck or expanding layers, this assert is in place \
            nn_arch should have only hidden layers -- no input and no output layer"

        need_drp = False
        if "dropouts" in kwargs:
            need_drp = True
            dropout = kwargs["dropouts"]

        self.model = nn.Sequential()

        prev = in_dim
        for idx, hdim in enumerate(nn_arch):
            self.model.add_module(f"{self.prefix}-emb_hid_{idx}", nn.Linear(prev, hdim))
            self.model.add_module(f"{self.prefix}-lReLU_{idx}", nn.LeakyReLU(inplace=True))
            if need_drp and dropout[idx] != 1:
                self.model.add_module(f"{self.prefix}-dropout_{idx}", nn.Dropout(p=dropout[idx]))
            prev = hdim
        
        self.model.add_module(f"{self.prefix}-last_layer", nn.Linear(prev, out_dim))
        
    def forward(self, x, beta):
        input = torch.cat((x, beta), dim=1)
        return self.model(input)

    def forward_r(self, x, beta):
        """Call this if the output layer needs to predict just one bit.
        """
        assert self.out_dim == 1, "If u need more than one output, why are u calling this?"
        return torch.sigmoid(self.forward(x, beta)).squeeze()

class ResNET(nn.Module):
    def __init__(self, out_dim, *args, **kwargs):
        super().__init__()
        self.out_dim = out_dim

        self.resnet_features =  tv_models.resnet18(pretrained=True)
        self.emb_dim = self.resnet_features.fc.in_features
        self.resnet_features.fc = nn.Linear(self.emb_dim, self.out_dim)

        self.sm = nn.Softmax(dim=1)

    def forward_proba(self, input):
        out = self.resnet_features(input)
        return self.sm(out)
    
    def forward(self, input):
        return self.resnet_features(input)
    
    def forward_labels(self, input):
        probs = self.forward_proba(input)
        probs, labels = torch.max(probs, dim=1)
        return labels


        