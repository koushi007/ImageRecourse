from turtle import forward
from numpy.core.defchararray import index
from sympy import Lambda
import torch.nn as nn
import torch
from torchvision import models as tv_models
import math
import numpy as np
import warnings
import utils.common_utils as cu
import our_method.constants as constants

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

class FNNXBeta(nn.Module):
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
        if "dropouts" in kwargs.keys():
            need_drp = True
            dropout = kwargs["dropouts"]
        self.batch_norm = False
        if constants.BATCH_NORM in kwargs.keys():
            self.batch_norm = kwargs[constants.BATCH_NORM]


        self.model = nn.Sequential()

        prev = in_dim
        for idx, hdim in enumerate(nn_arch):
            self.model.add_module(f"{self.prefix}-emb_hid_{idx}", nn.Linear(prev, hdim))
            self.model.add_module(f"{self.prefix}-lReLU_{idx}", nn.LeakyReLU(inplace=True))
            if need_drp and dropout[idx] != 1:
                self.model.add_module(f"{self.prefix}-dropout_{idx}", nn.Dropout(p=dropout[idx]))
            if self.batch_norm:
                self.model.add_module(f"{self.prefix}-bn_{idx}", nn.BatchNorm1d(hdim))
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
        if "dropouts" in kwargs.keys():
            need_drp = True
            dropout = kwargs["dropouts"]
        self.batch_norm = False
        if constants.BATCH_NORM in kwargs.keys():
            self.batch_norm = kwargs[constants.BATCH_NORM]

        self.model = nn.Sequential()

        prev = in_dim
        for idx, hdim in enumerate(nn_arch):
            self.model.add_module(f"{self.prefix}-emb_hid_{idx}", nn.Linear(prev, hdim))
            self.model.add_module(f"{self.prefix}-lReLU_{idx}", nn.LeakyReLU(inplace=True))
            if need_drp and dropout[idx] != 1:
                self.model.add_module(f"{self.prefix}-dropout_{idx}", nn.Dropout(p=dropout[idx]))
            if self.batch_norm:
                self.model.add_module(f"{self.prefix}-bn_{idx}", nn.BatchNorm1d(hdim))
            prev = hdim
        
        self.model.add_module(f"{self.prefix}-last_layer", nn.Linear(prev, out_dim))
        
    def forward(self, input):
        return self.model(input)



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



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, beta_dims, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.beta_dims = beta_dims
        self.max_len = np.prod(self.beta_dims)

        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(self.max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, beta: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        idx = torch.dot(beta, self.beta_dims)
        return self.dropout(self.pe[idx])

class BetaEmbedding(nn.Module):

    def __init__(self, d_model: int, beta_dim):
        super().__init__()
        self.beta_dims = beta_dim
        self.Emb = nn.Embedding(beta_dim, d_model)

    def forward(self, beta):
        return torch.squeeze(self.Emb(beta))



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()    
    def forward(self, x):
        return x

    
class ResNETRecourse(nn.Module):
    def __init__(self, out_dim, nn_arch, beta_dims, prefix, *args, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.beta_dims = beta_dims

        self.resnet_features =  tv_models.resnet18(pretrained=True)
        self.emb_dim = self.resnet_features.fc.in_features
        self.resnet_features.fc = Identity()

        self.beta_0_emb = BetaEmbedding(d_model=self.emb_dim, beta_dim=beta_dims[0])
        self.beta_1_emb = BetaEmbedding(d_model=self.emb_dim, beta_dim=beta_dims[1])
        self.beta_2_emb = BetaEmbedding(d_model=self.emb_dim, beta_dim=beta_dims[2])

        self.beta_classifier = FNN(in_dim=self.emb_dim, out_dim=nn_arch[-1], nn_arch=nn_arch[:-1], prefix=prefix, *args, **kwargs)

        self.xbeta_dim = nn_arch[-1]
        self.beta_0 = nn.Linear(self.xbeta_dim, self.beta_dims[0])
        self.beta_1 = nn.Linear(self.xbeta_dim, beta_dims[1])
        self.beta_2 = nn.Linear(self.xbeta_dim, beta_dims[2])

        self.sm = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

    def forward_proba(self, input, beta):
        beta0, beta1, beta2 = self.forward(input, beta)
        return torch.hstack([self.sm(beta0), self.sm(beta1), self.sm(beta2)])
    
    def forward(self, input, beta):
        x_emb = self.resnet_features(input)
        beta_emb = self.beta_0_emb(beta[:,0].view(-1, 1)) + self.beta_1_emb(beta[:,1].view(-1, 1)) + self.beta_2_emb(beta[:,2].view(-1, 1))
        beta_emb /= 3.

        xbeta_emb = x_emb + beta_emb
        xbeta_emb = self.beta_classifier(xbeta_emb)
        xbeta_emb = self.relu(xbeta_emb)

        beta0 = self.beta_0(xbeta_emb)
        beta1 = self.beta_1(xbeta_emb)
        beta2 = self.beta_2(xbeta_emb)

        return beta0, beta1, beta2

        
    def forward_labels(self, input, beta):
        beta0, beta1, beta2 = self.forward(input, beta)
        label = lambda b : torch.argmax(self.sm(b), dim=1).view(-1, 1)
        return torch.hstack([label(beta0), label(beta1), label(beta2)])


class ResNETPsi(nn.Module):
    def __init__(self, out_dim, nn_arch, beta_dims, prefix, *args, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.beta_dims = beta_dims

        self.resnet_features =  tv_models.resnet18(pretrained=True)
        self.emb_dim = self.resnet_features.fc.in_features
        self.resnet_features.fc = Identity()

        self.beta_0_emb = BetaEmbedding(d_model=self.emb_dim, beta_dim=beta_dims[0])
        self.beta_1_emb = BetaEmbedding(d_model=self.emb_dim, beta_dim=beta_dims[1])
        self.beta_2_emb = BetaEmbedding(d_model=self.emb_dim, beta_dim=beta_dims[2])

        self.r_classifier = FNN(in_dim=self.emb_dim, out_dim=nn_arch[-1], nn_arch=nn_arch[:-1], prefix=prefix, *args, **kwargs)

        # self.xbeta_dim = nn_arch[-1]
        # self.beta_0 = nn.Linear(self.xbeta_dim, self.beta_dims[0])
        # self.beta_1 = nn.Linear(self.xbeta_dim, beta_dims[1])
        # self.beta_2 = nn.Linear(self.xbeta_dim, beta_dims[2])

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, input, beta):
        x_emb = self.resnet_features(input)
        beta_emb = self.beta_0_emb(beta[:,0].view(-1, 1)) + self.beta_1_emb(beta[:,1].view(-1, 1)) + self.beta_2_emb(beta[:,2].view(-1, 1))
        beta_emb /= 3.

        xbeta_emb = x_emb + beta_emb
        r = self.r_classifier(xbeta_emb)
        
        return r
        
    def forward_r(self, input, beta):
        return self.sigmoid(self.forward(input, beta))