import random
import numpy as np
from tqdm import tqdm
import pickle as pkl
from pathlib import Path
import math
import copy
import warnings

#Use this function to generate data with 3 classes
def generate_data_3_class_dataset(d=10,
                                    r=5,
                                    total_data=6000,
                                    class_ratio_first=0.33,
                                    class_ratio_second=0.33,
                                    samples_per_instance=2,
                                    mu_first_list=None,
                                    mu_second_list=None,
                                    mu_third_list=None,
                                    sigma_first_list=None,
                                    sigma_second_list=None,
                                    sigma_third_list=None,
                                    data_type="train",
                                    dump_path=None):

    """Generates a three class dataset

    Args:
        d (int, optional): [description]. Defaults to 10. dimensionality of the dataset
        r (int, optional): [description]. Defaults to 5. Number of bits to **mask**
        total_data (int, optional): [description]. Defaults to 6000.
        class_ratio_first (float, optional): [description]. Defaults to 0.33.
        class_ratio_second (float, optional): [description]. Defaults to 0.33.
        samples_per_instance (int, optional): [description]. Defaults to 10. analogous to B_per_i 
        mu_first_list ([type], optional): [description]. Defaults to None.
        mu_second_list ([type], optional): [description]. Defaults to None.
        mu_third_list ([type], optional): [description]. Defaults to None.
        sigma_first_list ([type], optional): [description]. Defaults to None.
        sigma_second_list ([type], optional): [description]. Defaults to None.
        sigma_third_list ([type], optional): [description]. Defaults to None.
        data_type (str, optional): [description]. Defaults to "train".
        dump_path ([type], optional): [description]. Defaults to None.
    """

    x_f = []
    z_f = []
    b_f = []
    y_f = []
    instance_list_first = []

    
    x_s = []
    z_s = []
    b_s = []
    y_s = []
    instance_list_second = []

    x_t = []
    z_t = []
    b_t = []
    y_t = []
    instance_list_third = []

    instance_number = 0

    if mu_first_list is None:
        mu_first_list =  [0.1, 0.3, 0.1, 0.1,-0.3, 0.1, 0.1,-0.3, 0.1, 0.1]
        mu_second_list = [0.1,-0.3, 0.1, 0.1, 0.3, 0.1, 0.1,-0.3, 0.1, 0.1]
        mu_third_list =  [0.1,-0.3, 0.1, 0.1,-0.3, 0.1, 0.1, 0.3, 0.1, 0.1]

        sigma_first_list = [0.05   for i in range(d)]
        sigma_second_list = [0.05   for i in range(d)]
        sigma_third_list = [0.05   for i in range(d)]

    assert dump_path is not None, "Kindly provide a dump path"

    # data for first class y = 0
    for i in range(math.floor(total_data*class_ratio_first/samples_per_instance)):

        z = [0 for j in range(d)]
        # Sample z using the gaussian parameters
        for j in range(d):
            z[j] = random.gauss(mu_first_list[j],sigma_first_list[j])

        positions_list = [j for j in range(d)]

        # We are producing multiple samples(x,beta) for each instance (z)
        for k in range(samples_per_instance):

            x = [0 for j in range(d)]
            beta = [1 for j in range(d)]

            # select r elements to mask from the indexes list
            for j in random.sample(positions_list, r):
                beta[j] = 0

            # elementwise product to get x
            for j in range(d):
                x[j] = z[j]*beta[j]

            x_f.append(x)
            z_f.append(z)
            b_f.append(beta)
            y_f.append(0)
            instance_list_first.append(instance_number)

        instance_number = instance_number + 1



    # data for second class y = 1
    for i in range( math.floor(total_data*class_ratio_second/samples_per_instance)):

        z = [0 for j in range(d)]
        for j in range(d):
            z[j] = random.gauss(mu_second_list[j],sigma_second_list[j])

        positions_list = [j for j in range(d)]

        # We are producing multiple samples(x,beta) for each instance (z)
        for k in range(samples_per_instance):

            x = [0 for j in range(d)]
            beta = [1 for j in range(d)]

            # select r elements to mask from the indexes list
            for j in random.sample(positions_list, r):
                beta[j] = 0

            # elementwise product to get x
            for j in range(d):
                x[j] = z[j]*beta[j]

            x_s.append(x)
            z_s.append(z)
            b_s.append(beta)
            y_s.append(1)
            instance_list_second.append(instance_number)

        instance_number = instance_number + 1

    # data for third class y = 2
    for i in range( math.floor( (total_data-math.floor(total_data*class_ratio_first)-math.floor(total_data*class_ratio_second) ) / samples_per_instance ) ):

        z = [0 for j in range(d)]
        for j in range(d):
            z[j] = random.gauss(mu_third_list[j],sigma_third_list[j])

        positions_list = [j for j in range(d)]

        # We are producing multiple samples(x,beta) for each instance (z)
        for k in range(samples_per_instance):

            x = [0 for j in range(d)]
            beta = [1 for j in range(d)]

            # select r elements to mask from the indexes list
            for j in random.sample(positions_list, r):
                beta[j] = 0

            # elementwise product to get x
            for j in range(d):
                x[j] = z[j]*beta[j]

            x_t.append(x)
            z_t.append(z)
            b_t.append(beta)
            y_t.append(2)
            instance_list_third.append(instance_number)

        instance_number = instance_number + 1


    # dump the test and train data and make sure you get equal number of y = 0 and y = 1 and y = 2

    """
    This dumps a list of:
        x
        z
        beta
        labels
        instances
    """

    if data_type=="test" or data_type == "val":
        with open(dump_path, "wb") as file:

            pkl.dump([ x_f[:math.floor(total_data)][0::samples_per_instance] + x_s[:math.floor(total_data)][0::samples_per_instance] + x_t[:math.floor(total_data)][0::samples_per_instance], \
            z_f[:math.floor(total_data)][0::samples_per_instance] + z_s[:math.floor(total_data)][0::samples_per_instance] + z_t[:math.floor(total_data)][0::samples_per_instance], \
                b_f[:math.floor(total_data)][0::samples_per_instance] + b_s[:math.floor(total_data)][0::samples_per_instance] + b_t[:math.floor(total_data)][0::samples_per_instance], \
                    y_f[:math.floor(total_data)][0::samples_per_instance] + y_s[:math.floor(total_data)][0::samples_per_instance] + y_t[:math.floor(total_data)][0::samples_per_instance], \
                        instance_list_first[:math.floor(total_data)][0::samples_per_instance] + instance_list_second[:math.floor(total_data)][0::samples_per_instance] + instance_list_third[:math.floor(total_data)][0::samples_per_instance] ], file)

    if data_type=="train":
        with open(dump_path, "wb") as file:

            pkl.dump([ x_f[:math.floor(total_data)] + x_s[:math.floor(total_data)] + x_t[:math.floor(total_data)], \
            z_f[:math.floor(total_data)] + z_s[:math.floor(total_data)] + z_t[:math.floor(total_data)], \
                b_f[:math.floor(total_data)] + b_s[:math.floor(total_data)] + b_t[:math.floor(total_data)], \
                    y_f[:math.floor(total_data)] + y_s[:math.floor(total_data)] + y_t[:math.floor(total_data)], \
                        instance_list_first[:math.floor(total_data)] + instance_list_second[:math.floor(total_data)] + instance_list_third[:math.floor(total_data)]], file)

# This function will take the above generated pickle file(file_name) and generates the siblings list and dumps a new pickle file
# instance_list contaians the index for each data sample. 
# The sibling_list contains a list of indexes of the siblings of that data sample.
def process_train(file_name,dump_path):
    """This processes the generated Synthetic data and further adds derived data structs to the pickle

    Args:
        file_name ([type]): [description]
        dump_path ([type]): [description]
    
    Dumps in order:
        x_list 
        z_list
        beta_list
        label_list
        instance_list
        sibling_list
        R_list
        weights_list
        sij_list
    """

    # load the train data
    with open(file_name, "rb") as file:
        l = pkl.load(file)
    x_list, z_list, beta_list, label_list, instance_list = l[0], l[1], l[2], l[3], l[4]

    sibling_list = []
    siblings = [0]
    start  = 0

    for i in range(len(x_list)):

        if instance_list[i] == instance_list[i-1] and i!=0:
            siblings.append(i)
        elif instance_list[i] != instance_list[i-1] and i!=0:

            for j in range(i-start):
                sibling_list.append(siblings)

            siblings = [i]
            start = i

    for j in range(len(x_list)-start):
        sibling_list.append(siblings)

    #Here we are just initialising R_list,weights_list and sij_list
    R_list = [0.0 for i in range(len(x_list))]
    weights_list = [1.0 for i in range(len(x_list))]
    sij_list = copy.deepcopy(sibling_list)

    with open(dump_path, "wb") as file:

        pkl.dump([x_list, z_list, beta_list, label_list, instance_list, sibling_list, R_list, weights_list, sij_list], file)


if __name__ == "__main__":

    # %% Generate Training dataset and test datasets once and for all

    # Data generation config
    train_data = 6000

    # Note the test set will be divided by B_per_i. So pass accordingly
    test_data = 800
    val_data = 800

    syn_folder = Path("our_method/data/syn/B=2")
    syn_folder.mkdir(exist_ok=True, parents=True)

    train_path = syn_folder / "raw_train_3cls.pkl"
    test_path = syn_folder / "test_3cls.pkl"
    val_path = syn_folder / "val_3cls.pkl"
    process_train_path = syn_folder / "train_3cls.pkl"

    generate_data_3_class_dataset(total_data=train_data, data_type="train", dump_path=train_path)
    generate_data_3_class_dataset(total_data=test_data, data_type="test", dump_path=test_path)
    generate_data_3_class_dataset(total_data=val_data, data_type="val", dump_path=val_path)

    process_train(file_name=train_path, dump_path=process_train_path)


