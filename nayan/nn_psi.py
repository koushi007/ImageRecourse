from __future__ import print_function, division
import os
from random import shuffle
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import copy
from tqdm import tqdm
import pickle as pkl
from torch.utils.tensorboard import SummaryWriter
from config_new import config
from pathlib import Path
import matplotlib.pyplot as plt
import utils as utils
import constants
torch.manual_seed(0)
# %%
class PsiModel(nn.Module):
    def __init__(self,  layers, dropouts=None,  logger=None):
        """
        This is the class for psi model where we know a differential analytic formula for the transformation
        The architecture is Jury -> Layers -> Jury -> Logits
        Args:
            layers (list, optional): [number of neurons in each FCN]. Defaults to [5].
        """
        if logger is None:
            self.logger = utils.init_logger("logs/psi.logs")
        else:
            self.logger = logger 

        super(PsiModel, self).__init__()
        self.layers = layers
        self.need_dropout = dropouts is not None

        self.MODEL_NAME = "psi"

        self.writer = SummaryWriter(config[self.MODEL_NAME]["tblogs"])
       

        # Add dropout in the first layer if needed
        fc_inter = []
        if self.need_dropout:
            assert len(dropouts) == len(layers)+1, "Pass a dropout value for just after mobilenet as well"
            if dropouts[0] != 1.:
                self.logger.warning("Currently a droppot just after mobilenet will not take effect because in forwad I call eval() always on mobilenet")
                assert False, "This is not yet Implemented"
        if self.need_dropout and dropouts[1] != 1.:
            fc_inter.append(nn.Dropout(p = dropouts[1]))

        prev = 20
        # Implement Batch normalization also if needed
        for idx, num_nodes in enumerate(self.layers, start=1):
            fc_inter.append(nn.Linear(prev, num_nodes))
            # fc_inter.append(nn.ReLU(True))
            if self.need_dropout and dropouts[idx] != 1.:
                fc_inter.append(nn.Dropout(p=dropouts[idx]))
            prev = num_nodes
        fc_inter = nn.Sequential(*fc_inter)
        self.fc_inter = nn.Sequential(OrderedDict({
            "beta_predictor": fc_inter
        }))


    def forward(self, x, beta):
        """Runs forward on the entire model until the classifier

        Args:
            image ([type]): [description]
        """

        x = self.fc_inter(torch.cat((x,beta),1))
        
       
        return x
    
    def freeze(self):
        """Freezes the entire model
        """
        for param in self.parameters():
            param.requires_grad = False


    def get_trainable_params(self):
        """This returns the model's parameters for
        which we need to propagate the gradients

        Returns:
            List of parameters
        """
        trainable_params = []
        trainable_params = trainable_params + list(self.fc_inter.parameters())
        return trainable_params


    def train_model(self, config:dict, trainset, valset):
        """This function trains the psi model
        This works only with xm images. So set your dataloaders appropriately

        Args:
            config (dict): [description]
            trainset (SyntheticDataset): [description]
            valset (SyntheticDataset): [description]
        """
        since = time.time()
        
        valloader = DataLoader(valset, batch_size=config["psi"]["batch_size"])
        trainloader = DataLoader(trainset, batch_size=config["psi"]["batch_size"],
                                    num_workers=1, shuffle=True)
        dataset_sizes = {
            'train' : len(trainset),
            'val' : len(valset)
        }

        best_model_wts = copy.deepcopy(self.state_dict())
        best_loss = 1e6

        criterion_xent = nn.CrossEntropyLoss()
        criterion_mse = nn.MSELoss()
        criterion_bce = nn.BCELoss()

        writer = SummaryWriter(config['psi']["tblogs"])

        if config["psi"]["opt"] == "sgd":
            optimizer = optim.SGD(self.get_trainable_params(), lr=0.001, momentum=0.9)
        elif config["psi"]["opt"] == "adamw":
            optimizer = optim.AdamW(self.get_trainable_params())

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


       

        num_epochs = config["psi"]["epochs"]
        for epoch in range(num_epochs):
            if epoch==0:
                self.logger.info('Pre training phase')
            self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            self.logger.info('-' * 10)

            running_corrects = 0

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    # https://discuss.pytorch.org/t/how-does-batchnorm-keeps-track-of-running-mean/40084/6
                    loader = trainloader
                    
                    self.fc_inter.train()
                else:
                    loader = valloader
                    self.eval()   # Set model to evaluate mode

                running_loss = 0.0


                tqdm_bar = tqdm(range(len(loader)))
                for (tqdm_i, batch_data) in zip(tqdm_bar, loader):
                    x_list, beta_list, R_list = batch_data
                    
                    x = x_list.to(constants.device)
                    beta = beta_list.to(constants.device)
                    R = R_list.to(constants.device)

                    for i in range(1):

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forwards
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):

                            output_R = self.forward(x,beta)

                            output_R = torch.flatten(output_R)

                            # Using MSE loss
                            loss = criterion_mse(R , output_R)
                        
                                
                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                                self.writer.add_scalar(f'Loss/{self.MODEL_NAME}', loss.item(), \
                                                        epoch*len(loader)+tqdm_i)
                                self.writer.flush()
                                print("loss = ",loss)

                    # break

                    # statistics
                    running_loss += loss.item() * x.size(0)
                    running_corrects += torch.sum(R==(output_R>0.5))

                    # tqdm_bar.set_description(f'Epoch {epoch};')
                    tqdm_bar.update()


                # break

                if phase == 'train':
                    scheduler.step()
                    self.save_psi(file_name = f"model_epoch_{epoch}")

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = float(running_corrects) / dataset_sizes[phase]

                self.logger.info('{} Loss: {:.3f} Acc: {:.3f}'.format(
                                            phase, epoch_loss,epoch_acc))

                # deep copy the model
                # We do not have validation data so the below code basically selects the model which has the lowest training loss
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.state_dict())

                    self.save_psi()
                print()

        time_elapsed = time.time() - since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best val Loss: {:4f}'.format(best_loss))
        # load best model weights
        self.load_state_dict(best_model_wts)

    def save_psi(self, file_name = None):
        """Saves the model weights
        """
        dump_dir = Path(config["psi"]["dump_path"])
        dump_dir.mkdir(exist_ok=True, parents=True)

        if file_name is None:
            layers = config["psi"]["layers"]
            layers = [str(entry) for entry in layers]
            layers = "_".join(layers)
            file_prefix = config["psi"]["file_prefix"]
            file_name = f'{file_prefix}_psi_{layers}'
        
        torch.save(self.state_dict(),dump_dir / file_name)

    def load_psi(self, file_name=None):
        """Loads the psi specified in config if file name is none

        Args:
            file_name ([type], optional): [description]. Defaults to None.
        """
        if file_name is None:
            layers = config["psi"]["layers"]
            layers = [str(entry) for entry in layers]
            layers = "_".join(layers)
            file_prefix = config["psi"]["file_prefix"]
            file_name = constants.root_dir / Path(config["psi"]["load_path"]) / f'{file_prefix}_psi_{layers}'
        else:
            file_name = constants.root_dir / file_name
        self.logger.info(f"Loading psi from file path: {file_name}")
        self.load_state_dict(torch.load(file_name , map_location="cpu"))
        
    def test_psi(self,filename):
        """Tests the performance of the psi on test dataset.
        """
        
        global config

        with open("final_R_3_class_3000.pkl", 'rb') as file:
            r = pkl.load(file)

        R = r[0]    

        with open(filename, 'rb') as file:
            l = pkl.load(file)

        x_list, beta_list, label_list = l[0], l[2], l[3]

        test_ds = TensorDataset(torch.Tensor(x_list),torch.Tensor(beta_list),torch.Tensor(label_list),torch.Tensor(R))

        testloader = DataLoader(test_ds, batch_size=config["psi"]["batch_size"], num_workers=1, shuffle=False)

        it = iter(testloader)
        tqdm_bar = tqdm(range(len(testloader)))
        
        # set the model to eval mode
        self.eval()

        total_samples = 0

        total_corrects = 0

        for (tqdm_i, batch_data) in zip(tqdm_bar, testloader):
            x_list, beta_list, label_list, R_list = batch_data

            x = x_list.to(constants.device)
            beta = beta_list.to(constants.device)
            label = label_list.to(constants.device)

            output_R = self.forward(x,beta)

            output_R = (output_R>0.5).to(int)

            output_R = torch.flatten(output_R)

            corrects = torch.sum(R_list==output_R)

            total_corrects += corrects

            total_samples += len(label)
            tqdm_bar.update()

        percentage_acc = total_corrects / total_samples
        

        print("percent = ",percentage_acc*100)

        self.logger.info("Testing the psi")
        self.logger.info(f"percentage : {percentage_acc*100}")

    def output_R(self,load_path,output_R_path):
        """ This will output the R value for the test data (load_path) and dump the R value
        """
        
        global config
  

        with open(load_path, 'rb') as file:
            l = pkl.load(file)

        x_list, beta_list, label_list = l[0], l[2], l[3]

        test_ds = TensorDataset(torch.Tensor(x_list),torch.Tensor(beta_list),torch.Tensor(label_list))

        testloader = DataLoader(test_ds, batch_size=config["psi"]["batch_size"], num_workers=1, shuffle=False)

        it = iter(testloader)
        tqdm_bar = tqdm(range(len(testloader)))
        
        # set the model to eval mode
        self.eval()

        total_samples = 0

        output_R_list = []

        for (tqdm_i, batch_data) in zip(tqdm_bar, testloader):
            x_list, beta_list, label_list = batch_data

            x = x_list.to(constants.device)
            beta = beta_list.to(constants.device)
            label = label_list.to(constants.device)

            output_R = self.forward(x,beta)

            output_R = (output_R>0.5).to(int)

            output_R = torch.flatten(output_R)

            output_R_list.append(output_R)

            total_samples += len(label)
            tqdm_bar.update()

        output_R_list = torch.cat(output_R_list)

        with open(output_R_path, "wb") as file:
            pkl.dump([output_R_list], file)

        
if __name__ == "__main__":
# %%
    logger_file = config["logger_file"]
    assert "psi" in logger_file, f"check the logger file name {logger_file}"
    logger = utils.init_logger(f"logs/{logger_file}.log", file_mode="a")
    logger.info(utils.pretty_print(config))


    psi_layers = config["psi"]["layers"]
    dropouts = config["psi"]["dropouts"]
    psi = PsiModel(psi_layers, dropouts=dropouts, logger=logger)

    with open("processed_train_3_class.pkl", 'rb') as file:
        l = pkl.load(file)

    x_list, z_list, beta_list, label_list, instance_list, sibling_list, R_list, weights_list, sij_list = l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]


    with open("final_R_3_class_3000.pkl", 'rb') as file:
        r = pkl.load(file)

    R = r[0]    

    print(torch.sum(R))
    x_list = torch.Tensor(x_list)
    beta_list = torch.Tensor(beta_list)


    train_ds = TensorDataset(torch.Tensor(x_list),torch.Tensor(beta_list),R)

    psi.to(constants.device)
    psi.train_model(config, train_ds, train_ds)# trainset and valset
    logger.info("Completed training the psi")
    

# This is for testing
# %%
    psi.load_psi()
    psi.freeze()
    psi.eval()
    psi.test_psi("train_3_class.pkl")
