import torch
import torch.nn as nn

class FNN(nn.Module):
    """creates a Feed Forward Neural network with the specified Architecture
        nn ([type]): [description]
    """
    def __init__(self, in_dim, nn_arch, *args, **kwargs):
        """Creates a basic embedding block
        Args:
            in_dim ([type]): [description]
            embed_arch ([type]): [description]
        """
        super().__init__()

        if "prefix" in kwargs:
            self.prefix = f"{kwargs['prefix']}_" 
        else:
            self.prefix = ""

        self.model = nn.Sequential()

        prev = in_dim
        for idx, hdim in enumerate(nn_arch[:-1]):
            self.model.add_module(f"{self.prefix}emb_hid_{idx}", nn.Linear(prev, hdim))
            self.model.add_module("lReLU", nn.LeakyReLU(inplace=True))
            prev = hdim
        
        self.model.add_module(f"{self.prefix}last_layer", nn.Linear(prev, nn_arch[-1]))
        
    def forward(self, x, beta):
        input = torch.cat((x, beta), dim=1)
        return self.model(input)
    
