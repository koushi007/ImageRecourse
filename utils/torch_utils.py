import torch
from torch.utils.data.sampler import Sampler
import torch.utils.data as data_utils
import torch.nn as nn
import numpy as np
import random

def init_weights(m:nn.Module):
    def set_params(w):
        if isinstance(w, nn.Linear):
            torch.nn.init.xavier_uniform(w.weight)
            w.bias.data.fill_(0.01)
    m.apply(set_params)

def init_loader(data_ids, Z_ids, X, y, Z, Beta, shuffle=True, batch_size=None):
        T = torch.Tensor
        dataset = data_utils.TensorDataset(T(data_ids), T(Z_ids), T(X), T(y), T(Z), T(Beta))
        return data_utils.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def init_grp_loader(data_ids, Z_ids, X, y, Z, Beta, B_per_i, shuffle=True, batch_size=None):
    grp_arr = lambda arr : np.array(np.split(arr, int(len(arr) / B_per_i)))
    return init_loader(grp_arr(data_ids), grp_arr(Z_ids), grp_arr(X), grp_arr(y), grp_arr(Z), grp_arr(Beta), 
                            shuffle=shuffle, batch_size=int(batch_size / B_per_i))

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