from random import sample
import numpy as np

def sample_dataset(prior:float, 
                    B_per_i: int,
                    beta_noise:float, 
                    mean:dict, var:dict, 
                    dim: int,
                    num_train:int, num_test:int):
    """Samples a dataset according to the specified config and returns it

    Args:
        prior (float): class prior
        B_per_i (int): Number of betas we need to sample fro each example z_i. This is effective only for train data
        beta_noise (float): Number of betas that need to be OFF in each example
        mean (dict): mean for positive and negative classes. 
        var (dict): variance for positive and negative classes. 
        dim (int): dimensionality of the samples
        num_train (int): number of train samples
        num_test (int): number of test samples

    Returns:
        D_train: (z, beta, x, y)
        D_test: (z, beta, x, y)
    """
    pos_mean = mean["pos"]
    neg_mean = mean["neg"]
    pos_var = np.diag(var["pos"])
    neg_var = np.diag(var["neg"])

    total_samples = num_train+num_test

    Y = np.random.binomial(n=1, p=prior, size=total_samples) * 2 - 1
    Y = Y.astype(int)
    Y = np.sort(Y)

    num_pos = np.count_nonzero(Y==+1)
    num_neg = total_samples - num_pos
    
    num_beta_reqd = 0
    num_beta_reqd += num_train*B_per_i
    num_beta_reqd += num_test
    Beta = np.random.binomial(n=1, p=1-beta_noise, size=(num_beta_reqd*dim)).reshape(num_beta_reqd, dim)
    
    Z = np.concatenate((np.random.multivariate_normal(mean = pos_mean, cov=pos_var, size=num_pos), 
                        np.random.multivariate_normal(mean = neg_mean, cov=neg_var, size=num_neg)),
                        axis=0) 
    
    shuffle_idxs = np.random.permutation(np.arange(total_samples))
    Z, Y = Z[shuffle_idxs], Y[shuffle_idxs]
    
    Z_train, Z_test = Z[:num_train], Z[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]
    Beta_train, Beta_test = Beta[:num_train*B_per_i], Beta[num_train*B_per_i:]

    Z_train, Y_train = np.repeat(Z_train, repeats=B_per_i, axis=0), np.repeat(Y_train, repeats=B_per_i, axis=0)
    X_train = np.multiply(Z_train, Beta_train)
    X_test = np.multiply(Z_test, Beta_test)

    D_train = {
        "x": X_train,
        "beta": Beta_train,
        "z": Z_train,
        "y": Y_train
    }
    D_test = {
        "x": X_test,
        "beta": Beta_test,
        "z": Z_test,
        "y": Y_test
    }

    return D_train, D_test




# %% Unit Testing
# prior = 0.5
# beta_noise = 0.5
# B_per_i = 5
# mean = {
#     "pos": [0.1, 0.2, 0.3, 0.4],
#     "neg": [0.05, 0.15, 0.25, 0.35]
# }
# var = {
#     "pos": [0.1, 0.2, 0.3, 0.4],
#     "neg": [0.05, 0.15, 0.25, 0.35]
# }
# dim = 4
# num_train = 100
# num_test = 50
# D_train, D_test = sample_dataset(prior, B_per_i, beta_noise, mean, var, dim, num_train, num_test)

# assert D_train["x"].shape == (num_train*B_per_i, dim), "something wrong here! 1"
# assert D_train["y"].shape == (num_train*B_per_i,), "something wrong here! 2"
# assert D_test["x"].shape == (num_test, dim), "something wrong here! 3"
# assert D_test["y"].shape == (num_test, ), "something wrong here! 4"
# assert D_train["beta"].shape == (num_train*B_per_i, dim), "something wrong here! 5"



# %%