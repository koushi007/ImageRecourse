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
class PhiModel(nn.Module):
    def __init__(self,  layers, dropouts=None,  logger=None):
        """
        This is the class for phi model where we know a differential analytic formula for the transformation
        The architecture is Jury -> Layers -> Jury -> Logits
        Args:
            layers (list, optional): [number of neurons in each FCN]. Defaults to [5].
        """
        if logger is None:
            self.logger = utils.init_logger("logs/phi.logs")
        else:
            self.logger = logger 

        super(PhiModel, self).__init__()
        self.layers = layers
        self.need_dropout = dropouts is not None

        self.MODEL_NAME = "phi"

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
            
        """

        # The inputs are x and its corresponding beta
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
        """This function trains the phi model
        This works only with xm images. So set your dataloaders appropriately

        Args:
            config (dict): [description]
            trainset (SyntheticDataset): [description]
            valset (SyntheticDataset): [description]
        """
        since = time.time()
        
        valloader = DataLoader(valset, batch_size=config["phi"]["batch_size"])
        trainloader = DataLoader(trainset, batch_size=config["phi"]["batch_size"],
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

        writer = SummaryWriter(config['phi']["tblogs"])

        if config["phi"]["opt"] == "sgd":
            optimizer = optim.SGD(self.get_trainable_params(), lr=0.001, momentum=0.9)
        elif config["phi"]["opt"] == "adamw":
            optimizer = optim.AdamW(self.get_trainable_params())

        # Decay LR by a factor of 0.1 every 7 epochs
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


       

        num_epochs = config["phi"]["epochs"]
        for epoch in range(num_epochs):
            if epoch==0:
                self.logger.info('Pre training phase')
            self.logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
            self.logger.info('-' * 10)

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
                    x_list, beta_list, best_beta_list = batch_data
                    
                    x = x_list.to(constants.device)
                    beta = beta_list.to(constants.device)
                    best_beta = best_beta_list.to(constants.device)

                    for i in range(1):

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forwards
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):

                            # We will train it to output the best beta
                            output_beta = self.forward(x,beta)

                            # We are using MSE loss
                            loss = criterion_mse(best_beta , output_beta)
                        
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

                    # tqdm_bar.set_description(f'Epoch {epoch};')
                    tqdm_bar.update()


                # break

                if phase == 'train':
                    scheduler.step()
                    self.save_phi(file_name = f"model_epoch_{epoch}")

                epoch_loss = running_loss / dataset_sizes[phase]
                # epoch_acc = float(running_corrects) / dataset_sizes[phase]

                self.logger.info('{} Loss: {:.3f}'.format(
                                            phase, epoch_loss))

                # deep copy the model
                # We do not have validation data so the below code basically selects the model which has the lowest training loss
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(self.state_dict())

                    self.save_phi()
                print()

        time_elapsed = time.time() - since
        self.logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        self.logger.info('Best val Loss: {:4f}'.format(best_loss))
        # load best model weights
        self.load_state_dict(best_model_wts)

    def save_phi(self, file_name = None):
        """Saves the model weights
        """
        dump_dir = Path(config["phi"]["dump_path"])
        dump_dir.mkdir(exist_ok=True, parents=True)

        if file_name is None:
            layers = config["phi"]["layers"]
            layers = [str(entry) for entry in layers]
            layers = "_".join(layers)
            file_prefix = config["phi"]["file_prefix"]
            file_name = f'{file_prefix}_phi_{layers}'
        
        torch.save(self.state_dict(),dump_dir / file_name)

    def load_phi(self, file_name=None):
        """Loads the phi specified in config if file name is none

        Args:
            file_name ([type], optional): [description]. Defaults to None.
        """
        if file_name is None:
            layers = config["phi"]["layers"]
            layers = [str(entry) for entry in layers]
            layers = "_".join(layers)
            file_prefix = config["phi"]["file_prefix"]
            file_name = constants.root_dir / Path(config["phi"]["load_path"]) / f'{file_prefix}_phi_{layers}'
        else:
            file_name = constants.root_dir / file_name
        self.logger.info(f"Loading phi from file path: {file_name}")
        self.load_state_dict(torch.load(file_name , map_location="cpu"))
        
    def test_phi(self,filename):
        """Tests the performance of the phi on test dataset.
        """
        
        global config

        with open(filename, 'rb') as file:
            l = pkl.load(file)

        x_list, z_list, beta_list, label_list = l[0], l[1], l[2], l[3]

        test_ds = TensorDataset(torch.Tensor(x_list),torch.Tensor(beta_list),torch.Tensor(label_list),torch.Tensor(z_list))

        testloader = DataLoader(test_ds, batch_size=config["phi"]["batch_size"], num_workers=1, shuffle=False)

        it = iter(testloader)
        tqdm_bar = tqdm(range(len(testloader)))
        
        # set the model to eval mode
        self.eval()

        total_samples = 0

        # this array will store the number of times the beta takes the value 1 for all the 10 betas
        total_beta_equal_1 = torch.tensor([0 for i in range(len(x_list[0]))]).to(constants.device)

        output_beta_list = []

        for (tqdm_i, batch_data) in zip(tqdm_bar, testloader):
            x_list, beta_list, label_list, z_list = batch_data

            x = x_list.to(constants.device)
            beta = beta_list.to(constants.device)
            label = label_list.to(constants.device)
            z = z_list.to(constants.device)

            output_beta = self.forward(x,beta)

            # If output is greater than 0.5 then we consider it as 1
            output_beta = (output_beta>0.5).to(int)

            beta_equal_1 = torch.sum(output_beta,axis=0)

            total_beta_equal_1 += beta_equal_1

            output_beta_list.append(output_beta)

            total_samples += len(label)
            tqdm_bar.update()

        beta_equal_1_percentage = total_beta_equal_1 / total_samples

        print("percentage time the beta is 1 = ",beta_equal_1_percentage*100)

        self.logger.info("Testing the phi")

        self.logger.info(f"beta_equal_1 percentage : {beta_equal_1_percentage}")

    def output_beta(self,load_path,output_R_path,w1,b1,w2,b2,w3,b3,num_beta_to_unmask):
        """ This will output the beta values for the test data(load_path)
        It will output the beta values only for the data which has R = 1 (output_R_path contains the R values which are obtained from the output of nn psi)
        It will also give the accuracy before and after recoursse
        """
        
        global config

        with open(load_path, 'rb') as file:
            l = pkl.load(file)

        x_list, z_list, beta_list, label_list = l[0], l[1], l[2], l[3]

        # To find the accuracy without any recourse
        first_test = torch.mm(torch.Tensor(x_list), w1) + b1
        second_test = torch.mm(torch.Tensor(x_list), w2) + b2
        third_test = torch.mm(torch.Tensor(x_list), w3) + b3
                
        y_hat_test = torch.softmax(torch.cat([first_test,second_test,third_test],axis = 1),dim=1)
        test_acc_without_recourse = torch.sum(torch.Tensor(label_list) == (torch.argmax(y_hat_test,axis =1).to(int)) ) / len(label_list)



        with open(output_R_path, 'rb') as file:
            output_R = pkl.load(file)
        output_R = output_R[0]

        test_ds = TensorDataset(torch.Tensor(x_list)[output_R==1],torch.Tensor(beta_list)[output_R==1],torch.Tensor(label_list)[output_R==1],torch.Tensor(z_list)[output_R==1])

        testloader = DataLoader(test_ds, batch_size=config["phi"]["batch_size"], num_workers=1, shuffle=False)

        tqdm_bar = tqdm(range(len(testloader)))
        
        # set the model to eval mode
        self.eval()

        total_samples = 0
        total_beta_equal_1 = torch.tensor([0 for i in range(len(x_list[0]))]).to(constants.device)

        output_beta_list = []

        x_new_list = []

        for (tqdm_i, batch_data) in zip(tqdm_bar, testloader):
            x_list_batch, beta_list_batch, label_list_batch, z_list_batch = batch_data

            x_new = x_list_batch

            x = x_list_batch.to(constants.device)
            beta = beta_list_batch.to(constants.device)
            label = label_list_batch.to(constants.device)
            z = z_list_batch#.to(constants.device)

            output_beta = self.forward(x,beta)

            # Select the top k beta and unmask them
            top_beta = torch.topk(output_beta,k=num_beta_to_unmask,axis = 1).indices

            for i in range(len(x)):
                for j in top_beta[i]:
                    x_new[i][j] = z[i][j]


            output_beta = (output_beta>0.5).to(int)

            beta_equal_1 = torch.sum(output_beta,axis=0)

            total_beta_equal_1 += beta_equal_1

            output_beta_list.append(output_beta)

            x_new_list.append(x_new)

            total_samples += len(label)
            tqdm_bar.update()


        x_new_list = torch.cat([torch.cat(x_new_list),torch.Tensor(x_list)[output_R==0]])
        label_new_list = torch.cat([torch.Tensor(label_list)[output_R==1],torch.Tensor(label_list)[output_R==0]])

        # To find the accuracy with recourse
        first_test_new = torch.mm(torch.Tensor(x_new_list), w1) + b1
        second_test_new = torch.mm(torch.Tensor(x_new_list), w2) + b2
        third_test_new = torch.mm(torch.Tensor(x_new_list), w3) + b3
                
        y_hat_test_new = torch.softmax(torch.cat([first_test_new,second_test_new,third_test_new],axis = 1),dim=1)
        test_acc_with_recourse = torch.sum(torch.Tensor(label_new_list) == (torch.argmax(y_hat_test_new,axis =1).to(int)) ) / len(label_new_list)

        print("before accuracay = ",test_acc_without_recourse, "after accuracy = ",test_acc_with_recourse)

        beta_equal_1_percentage = total_beta_equal_1 / total_samples

        print("percent of times beta is predicted as 1 = ",beta_equal_1_percentage*100)
        print("done")


       
if __name__ == "__main__":
# %%
    logger_file = config["logger_file"]
    assert "phi" in logger_file, f"check the logger file name {logger_file}"
    logger = utils.init_logger(f"logs/{logger_file}.log", file_mode="a")
    logger.info(utils.pretty_print(config))

    phi_layers = config["phi"]["layers"]
    dropouts = config["phi"]["dropouts"]
    phi = PhiModel(phi_layers, dropouts=dropouts, logger=logger)

    with open("processed_train_3_class.pkl", 'rb') as file:
        l = pkl.load(file)

    x_list, z_list, beta_list, label_list, instance_list, sibling_list, R_list, weights_list, sij_list = l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]


    with open("final_R_3_class_3000.pkl", 'rb') as file:
        r = pkl.load(file)


    R = r[0]    
    w1 = r[1]
    b1 = r[2]
    w2 = r[3]
    b2 = r[4]
    w3 = r[5]
    b3 = r[6]

    _, sij_list = utils.find_best_sij(R,w1,b1,w2,b2,w3,b3,label_list,process_train_path = "processed_train_3_class.pkl")


    # print(torch.sum(R))
    x_list = torch.Tensor(x_list)
    beta_list = torch.Tensor(beta_list)
    sij_list = torch.stack(sij_list)
    sibling_list = torch.IntTensor(sibling_list)
    label_list = torch.Tensor(label_list)

    x = x_list[R==1]
    beta = beta_list[R==1]
    sij = sij_list[R==1]
    siblings = sibling_list[R==1]
    label = label_list[R==1]

    x_list_new = []
    beta_list_new = []
    best_beta_list = []
    label_list_new = []

    # Prepare dataset to train the NN phi. best_beta_list is the list of betas of the Sij's
    for i in range(x.shape[0]):
        
        for j in range(sij[i].shape[0]):

            if sij[i][j] != 0:

                x_list_new.append(x[i])
                beta_list_new.append(beta[i])
                best_beta_list.append(beta_list[siblings[i][j]])
                label_list_new.append(label[i])

    train_ds = TensorDataset(torch.stack(x_list_new),torch.stack(beta_list_new),torch.stack(best_beta_list))

    phi.to(constants.device)

    phi.train_model(config, train_ds, train_ds)
    logger.info("Completed training the phi")
    

# This is for testing
# %%
    phi.load_phi()
    phi.freeze()
    phi.eval()
    phi.test_phi("train_3_class.pkl")
