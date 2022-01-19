import torch
from torch.utils.data.sampler import Sampler
import torch.utils.data as data_utils
import torch.nn as nn
import numpy as np
import random
from our_method.models import ResNET
import our_method.constants as constants
import torch.optim as optim

def init_weights(m:nn.Module):

    assert isinstance(m, ResNET) == False, "Why the hell are u initializing the weights of a pretrained model."

    def set_params(w):
        if isinstance(w, nn.Linear):
            torch.nn.init.xavier_uniform(w.weight)
            w.bias.data.fill_(0.01)
    m.apply(set_params)

def init_loader(data_ids, Z_ids, X, y, Z, Beta, shuffle=True, batch_size=None, **kwargs):
        T = torch.Tensor
        dataset = CustomTensorDataset(T(data_ids), T(Z_ids), T(X), T(y), T(Z), T(Beta), **kwargs)
        return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def init_grp_loader(data_ids, Z_ids, X, y, Z, Beta, B_per_i, shuffle=True, batch_size=None, **kwargs):
    grp_arr = lambda arr : np.array(np.split(arr, int(len(arr) / B_per_i)))
    return init_loader(grp_arr(data_ids), grp_arr(Z_ids), grp_arr(X), grp_arr(y), grp_arr(Z), grp_arr(Beta), 
                            shuffle=shuffle, batch_size=int(batch_size / B_per_i), **kwargs)

def generic_init_loader(*args, **kwargs):
    """This is a generic init loader. We just create a dataset of any which crap u send to us
    Use the above init_loader to ensure some discipline.
    But pls pass batch size and shuffle as keargs in the very least
    Or pass a batch sampler if u wish to perform custom batching
    """
    T = torch.Tensor
    dataset = data_utils.TensorDataset(*[T(entry) for entry in args])
    if "sampler" in kwargs:
        return data_utils.DataLoader(dataset, batch_size=kwargs["batch_size"], sampler=kwargs["sampler"])
    else:
        shuffle = kwargs["shuffle"]
        batch_size = kwargs["batch_size"]
        return data_utils.DataLoader(dataset, shuffle=shuffle, batch_size=batch_size)


class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, labels, indices=None, class_choice="least_sampled"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
            class_choice: a string indicating how class will be selected for every
            sample:
                "least_sampled": class with the least number of sampled labels so far
                "random": class is chosen uniformly at random
                "cycle": the sampler cycles through the classes sequentially
        """
        self.labels = labels
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(labels))

        self.num_classes = self.labels.shape[1]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        return class_

    def __len__(self):
        return len(self.indices)


class CustomTensorDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, data_ids, Z_ids, X, y, Z, Beta, *args, **kwargs):
        self.data_ids = data_ids
        self.Z_ids = Z_ids
        self.X = X
        self.y = y
        self.Z = Z
        self.Beta = Beta

        self.transform = None
        if constants.TRANSFORM in kwargs:
            self.transform = kwargs[constants.TRANSFORM]

    def __getitem__(self, index):
        data_id, z_id, x, y, z, beta = self.data_ids[index], self.Z_ids[index], self.X[index], self.y[index], self.Z[index], self.Beta[index]
        if self.transform is not None:
            x, z = self.transform(x), self.transform(z)
        return data_id, z_id, x, y, z, beta

    def __len__(self):
        return self.X.shape[0]

class CustomPhiDataset(data_utils.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, R_ids, X, Beta, Sib_beta, Sij, Sib_losses, *args,  **kwargs):
        self.R_ids = R_ids
        self.X = X
        self.Beta = Beta
        self.Sib_beta = Sib_beta
        self.Sij = Sij
        self.Sib_losses = Sib_losses

        self.transform = None
        if constants.TRANSFORM in kwargs:
            self.transform = kwargs[constants.TRANSFORM]

    def __getitem__(self, index):
        r_id, x, beta, sib_beta, sij, sib_losses = self.R_ids[index], self.X[index], self.Beta[index], \
                        self.Sib_beta[index], self.Sij[index], self.Sib_losses[index]

        if self.transform is not None:
            x = self.transform(x)
        return r_id, x, beta, sib_beta, sij, sib_losses

    def __len__(self):
        return len(self.R_ids)






def get_lr_scheduler(optimizer, scheduler_name, n_rounds=None):
    """
    Gets torch.optim.lr_scheduler given an lr_scheduler name and an optimizer
    :param optimizer:
    :type optimizer: torch.optim.Optimizer
    :param scheduler_name: possible are
    :type scheduler_name: str
    :param n_rounds: number of training rounds, only used if `scheduler_name == multi_step`
    :type n_rounds: int
    :return: torch.optim.lr_scheduler
    """

    if scheduler_name == "sqrt":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / np.sqrt(x) if x > 0 else 1)

    elif scheduler_name == "linear":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1 / x if x > 0 else 1)

    elif scheduler_name == "constant":
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 1)

    elif scheduler_name == "cosine_annealing":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200, eta_min=0)

    elif scheduler_name == "multi_step":
        assert n_rounds is not None, "Number of rounds is needed for \"multi_step\" scheduler!"
        milestones = [n_rounds//2, 3*(n_rounds//4)]
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    else:
        raise NotImplementedError("Other learning rate schedulers are not implemented")


def get_loader_subset(loader:data_utils.DataLoader, subset_idxs:list, shuffle=False):
    """Returns a data loader with the mentioned subset indices
    """
    subset_ds = data_utils.Subset(dataset=loader.dataset, indices=subset_idxs)
    return data_utils.DataLoader(subset_ds, batch_size=loader.batch_size, shuffle=shuffle)




