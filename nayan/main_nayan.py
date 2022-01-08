from functools import reduce
import random
import numpy as np
from tqdm import tqdm
import pickle as pkl
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import utils as utils
import sys
import math
from sklearn import linear_model
import copy
from logistic_regression import *
from nn_phi import PhiModel
from nn_psi import PsiModel
from config_new import config
import constants
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.set_printoptions(precision=5)

#Use this function to generate data with 2 classes, this is used only to plot the accuracy graphs
def generate_data_2_class(test_ratio,d,r,total_data,class_ratio,samples_per_instance,mean_step_size):
    
    mu_first_list = [mean_step_size,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    mu_second_list = [-1*mean_step_size,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.1]
    sigma_first_list = [0.05   for i in range(d)]
    sigma_second_list = [0.05   for i in range(d)]

    x_list_first = []
    z_list_first = []
    beta_list_first = []
    label_list_first = []
    instance_list_first = []

    x_list_second = []
    z_list_second = []
    beta_list_second = []
    label_list_second = []
    instance_list_second = []

    instance_number = 0

    # data for first class y = +1
    for i in range(math.floor(total_data*class_ratio/samples_per_instance)):

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

            x_list_first.append(x)
            z_list_first.append(z)
            beta_list_first.append(beta)
            label_list_first.append(1)
            instance_list_first.append(instance_number)

        instance_number = instance_number + 1



    # data for first class y = -1
    for i in range( math.floor( (total_data-math.floor(total_data*class_ratio)) / samples_per_instance ) ):

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

            x_list_second.append(x)
            z_list_second.append(z)
            beta_list_second.append(beta)
            label_list_second.append(0)
            instance_list_second.append(instance_number)

        instance_number = instance_number + 1

    # dump the test and train data and make sure you get equal number of y = +1 and y = -1
    with open("test_2_class.pkl", "wb") as file:
      
        pkl.dump([ x_list_first[:math.floor(total_data*class_ratio*test_ratio)][0::samples_per_instance] + x_list_second[:math.floor(total_data*class_ratio*test_ratio)][0::samples_per_instance], \
         z_list_first[:math.floor(total_data*class_ratio*test_ratio)][0::samples_per_instance] + z_list_second[:math.floor(total_data*class_ratio*test_ratio)][0::samples_per_instance], \
             beta_list_first[:math.floor(total_data*class_ratio*test_ratio)][0::samples_per_instance] + beta_list_second[:math.floor(total_data*class_ratio*test_ratio)][0::samples_per_instance], \
                 label_list_first[:math.floor(total_data*class_ratio*test_ratio)][0::samples_per_instance] + label_list_second[:math.floor(total_data*class_ratio*test_ratio)][0::samples_per_instance], \
                     instance_list_first[:math.floor(total_data*class_ratio*test_ratio)][0::samples_per_instance] + instance_list_second[:math.floor(total_data*class_ratio*test_ratio)][0::samples_per_instance]], file)


    with open("train_2_class.pkl", "wb") as file:

        pkl.dump([ x_list_first[math.floor(total_data*class_ratio*test_ratio):] + x_list_second[math.floor(total_data*class_ratio*test_ratio):], \
         z_list_first[math.floor(total_data*class_ratio*test_ratio):] + z_list_second[math.floor(total_data*class_ratio*test_ratio):], \
             beta_list_first[math.floor(total_data*class_ratio*test_ratio):] + beta_list_second[math.floor(total_data*class_ratio*test_ratio):], \
                 label_list_first[math.floor(total_data*class_ratio*test_ratio):] + label_list_second[math.floor(total_data*class_ratio*test_ratio):], \
                     instance_list_first[math.floor(total_data*class_ratio*test_ratio):] + instance_list_second[math.floor(total_data*class_ratio*test_ratio):]], file)

#Use this function to generate data with 3 classes
def generate_data_3_class_dataset(d,r,total_data,class_ratio_first,class_ratio_second,class_ratio_third,samples_per_instance,mean_step_size,mu_first_list,mu_second_list,mu_third_list,sigma_first_list,sigma_second_list,sigma_third_list,data_type,dump_path):


    x_list_first = []
    z_list_first = []
    beta_list_first = []
    label_list_first = []
    instance_list_first = []

    
    x_list_second = []
    z_list_second = []
    beta_list_second = []
    label_list_second = []
    instance_list_second = []

    x_list_third = []
    z_list_third = []
    beta_list_third = []
    label_list_third = []
    instance_list_third = []

    instance_number = 0

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

            x_list_first.append(x)
            z_list_first.append(z)
            beta_list_first.append(beta)
            label_list_first.append(0)
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

            x_list_second.append(x)
            z_list_second.append(z)
            beta_list_second.append(beta)
            label_list_second.append(1)
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

            x_list_third.append(x)
            z_list_third.append(z)
            beta_list_third.append(beta)
            label_list_third.append(2)
            instance_list_third.append(instance_number)

        instance_number = instance_number + 1


    # dump the test and train data and make sure you get equal number of y = 0 and y = 1 and y = 2
    if data_type=="test" or data_type == "val":
        with open(dump_path, "wb") as file:

            pkl.dump([ x_list_first[:math.floor(total_data)][0::samples_per_instance] + x_list_second[:math.floor(total_data)][0::samples_per_instance] + x_list_third[:math.floor(total_data)][0::samples_per_instance], \
            z_list_first[:math.floor(total_data)][0::samples_per_instance] + z_list_second[:math.floor(total_data)][0::samples_per_instance] + z_list_third[:math.floor(total_data)][0::samples_per_instance], \
                beta_list_first[:math.floor(total_data)][0::samples_per_instance] + beta_list_second[:math.floor(total_data)][0::samples_per_instance] + beta_list_third[:math.floor(total_data)][0::samples_per_instance], \
                    label_list_first[:math.floor(total_data)][0::samples_per_instance] + label_list_second[:math.floor(total_data)][0::samples_per_instance] + label_list_third[:math.floor(total_data)][0::samples_per_instance], \
                        instance_list_first[:math.floor(total_data)][0::samples_per_instance] + instance_list_second[:math.floor(total_data)][0::samples_per_instance] + instance_list_third[:math.floor(total_data)][0::samples_per_instance] ], file)

    if data_type=="train":
        with open(dump_path, "wb") as file:

            pkl.dump([ x_list_first[:math.floor(total_data)] + x_list_second[:math.floor(total_data)] + x_list_third[:math.floor(total_data)], \
            z_list_first[:math.floor(total_data)] + z_list_second[:math.floor(total_data)] + z_list_third[:math.floor(total_data)], \
                beta_list_first[:math.floor(total_data)] + beta_list_second[:math.floor(total_data)] + beta_list_third[:math.floor(total_data)], \
                    label_list_first[:math.floor(total_data)] + label_list_second[:math.floor(total_data)] + label_list_third[:math.floor(total_data)], \
                        instance_list_first[:math.floor(total_data)] + instance_list_second[:math.floor(total_data)] + instance_list_third[:math.floor(total_data)]], file)

# This function will take the above generated pickle file(file_name) and generates the siblings list and dumps a new pickle file
# instance_list contaians the index for each data sample. 
# The sibling_list contains a list of indexes of the siblings of that data sample.
def process_train(file_name,dump_path):

    # load the train data
    with open(file_name, "rb") as file:
        l = pkl.load(file)
    x_list, z_list, beta_list, label_list, instance_list = l[0], l[1], l[2], l[3], l[4]

    sibling_list = []
    prev = -1
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


# This function will take the path to the processed pickle file and selects the elements into R and dumps the R array(R_dump_path)
# num_R is the number of elements you want to selct into R
def recourse(w1,b1,w2,b2,w3,b3,process_train_path,R_dump_path,test_file_name,num_R,num_randomly_sample_indices,num_grad_descent_iterations):

    with open(process_train_path, "rb") as file:
        l = pkl.load(file)
    x_list, z_list, beta_list, label_list, instance_list, sibling_list, R_list, weights_list, sij_list = l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]
    
    with open(test_file_name, 'rb') as file:
        l_test = pkl.load(file)
    x_list_test, z_list_test, beta_list_test, label_list_test = l_test[0], l_test[1], l_test[2], l_test[3]

    x_list = torch.Tensor(x_list)
    R_list = torch.Tensor(R_list)
    y = torch.LongTensor(label_list)
    sibling_list = torch.LongTensor(sibling_list)

    index_list = [i for i in range(x_list.shape[0])]

    for t in range(num_R):

        new_weights_list, new_sij_list = utils.find_sij(R_list,w1,b1,w2,b2,w3,b3,y,process_train_path)

        new_sij_list = torch.stack(new_sij_list)

        w1_list = []
        b1_list = []
        w2_list = []
        b2_list = []
        w3_list = []
        b3_list = []
        loss_list = []
        k_index_list = []

        w1_initial = copy.deepcopy(w1)
        b1_initial = copy.deepcopy(b1)
        w2_initial = copy.deepcopy(w2)
        b2_initial = copy.deepcopy(b2)
        w3_initial = copy.deepcopy(w3)
        b3_initial = copy.deepcopy(b3)

        w1_initial.requires_grad = False
        b1_initial.requires_grad = False
        w2_initial.requires_grad = False
        b2_initial.requires_grad = False
        w3_initial.requires_grad = False
        b3_initial.requires_grad = False

        # Since searching for the minimum loss for all indexes is difficult, we randomly sample some of the indices "num_randomly_sample_indices"
        for k in random.sample(index_list, num_randomly_sample_indices):

            # If it is alreaady selected in R then skip it
            if R_list[k]==1:
                continue

            w1 = copy.deepcopy(w1_initial)
            b1 = copy.deepcopy(b1_initial)
            w2 = copy.deepcopy(w2_initial)
            b2 = copy.deepcopy(b2_initial)
            w3 = copy.deepcopy(w3_initial)
            b3 = copy.deepcopy(b3_initial)


            w1.requires_grad = True
            b1.requires_grad = True
            w2.requires_grad = True
            b2.requires_grad = True
            w3.requires_grad = True
            b3.requires_grad = True

            optimizer = optim.SGD([w1,b1,w2,b2,w3,b3], lr=0.1, momentum=0.0)


            # For adding an element to R, we decrease the weight of the sample and distribute it evenly among the Sij's
            new_weights_list[k] = new_weights_list[k] - 1

            if torch.sum(new_sij_list[k]) == 0.0:
                # If there are no Sij's 
                new_weights_list[k] = new_weights_list[k] + 1.0
            else:
                # If there are Sij's then distribute the weights
                new_weights_list[sibling_list[k]] = new_weights_list[sibling_list[k]] + new_sij_list[k]/(torch.sum(new_sij_list[k])+1e-8)


            new_weights_list = torch.Tensor(new_weights_list)

            # run gradient descent for "num_grad_descent_iterations"
            for epoch in range(num_grad_descent_iterations):

                optimizer.zero_grad()

                first = torch.mm(x_list, w1) + b1
                second = torch.mm(x_list, w2) + b2
                third = torch.mm(x_list, w3) + b3
                
                y_hat = torch.softmax(torch.cat([first,second,third],axis = 1),dim=1)

                loss_unweighted = torch.nn.CrossEntropyLoss(reduce=False)(y_hat,y)

                loss = torch.mean(loss_unweighted*new_weights_list) 


                loss.backward()
                optimizer.step()

            # Store the all the weights and finally select the weights which correspond to the minimum loss
            w1_list.append(copy.deepcopy(w1))
            b1_list.append(copy.deepcopy(b1))
            w2_list.append(copy.deepcopy(w2))
            b2_list.append(copy.deepcopy(b2))
            w3_list.append(copy.deepcopy(w3))
            b3_list.append(copy.deepcopy(b3))
            loss_list.append(loss)
            k_index_list.append(k)
            

            new_weights_list[k] = new_weights_list[k] + 1
            if torch.sum(new_sij_list[k]) == 0.0:
                new_weights_list[k] = new_weights_list[k] - 1.0
            else:
                new_weights_list[sibling_list[k]] = new_weights_list[sibling_list[k]] - new_sij_list[k]/(torch.sum(new_sij_list[k])+1e-8)


        # Find the index which corresponds to the minimum loss and add it to R
        index = torch.argmin(torch.Tensor(loss_list))
        w1 = w1_list[index]
        b1 = b1_list[index]
        w2 = w2_list[index]
        b2 = b2_list[index]
        w3 = w3_list[index]
        b3 = b3_list[index]
        R_list[k_index_list[index]] = 1


        # find the test accuracy at each step to make sure it is not overfitting on the train data
        first_test = torch.mm(torch.Tensor(x_list_test), w1) + b1
        second_test = torch.mm(torch.Tensor(x_list_test), w2) + b2
        third_test = torch.mm(torch.Tensor(x_list_test), w3) + b3
        y_hat_test = torch.softmax(torch.cat([first_test,second_test,third_test],axis = 1),dim=1)
        test_acc = torch.sum(torch.Tensor(label_list_test) == (torch.argmax(y_hat_test,axis =1).to(int)) ) / len(label_list_test)

        print("minimum loss found is = ",loss_list[index],"test acc = ",test_acc,"iter number = ",t)
    
    
    print("number of R selected = ",torch.sum(R_list))

    with open(R_dump_path, "wb") as file:
        pkl.dump([R_list,w1,b1,w2,b2,w3,b3], file)


    return
   
# This function will do the gradient descent on the whole dataset
# file name is the patj to dataset, w_b_file is the path where the weights are dumped
def grad_descent_torch(file_name,test_file_name,w_b_file):

    with open(file_name, "rb") as file:
        l = pkl.load(file)
    x_list, z_list, beta_list, label_list, instance_list, sibling_list, R_list, weights_list, sij_list = l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8]
 
    with open(test_file_name, 'rb') as file:
        l_test = pkl.load(file)

    x_list_test, z_list_test, beta_list_test, label_list_test = l_test[0], l_test[1], l_test[2], l_test[3]


    w1 = torch.zeros((len(x_list[0]),1))
    b1 = torch.tensor(0.0)
    w2 = torch.zeros((len(x_list[0]),1))
    b2 = torch.tensor(0.0)
    w3 = torch.zeros((len(x_list[0]),1))
    b3 = torch.tensor(0.0)

    w1.requires_grad = True
    b1.requires_grad = True
    w2.requires_grad = True
    b2.requires_grad = True
    w3.requires_grad = True
    b3.requires_grad = True

    optimizer = optim.SGD([w1,b1,w2,b2,w3,b3], lr=0.01, momentum=0.0)

    x_list = torch.Tensor(x_list)
    R_list = torch.Tensor(R_list)
    y = torch.LongTensor(label_list)

    for epoch in range(10):

        optimizer.zero_grad()

        first = torch.mm(x_list, w1) + b1
        second = torch.mm(x_list, w2) + b2
        third = torch.mm(x_list, w3) + b3
        
        y_hat = torch.softmax(torch.cat([first,second,third],axis = 1),dim=1)
        loss = torch.nn.CrossEntropyLoss()(y_hat,y)

        w1_prev = copy.deepcopy(w1)
        w2_prev = copy.deepcopy(w2)
        w3_prev = copy.deepcopy(w3)

        loss.backward()
        optimizer.step()

        first_test = torch.mm(torch.Tensor(x_list_test), w1) + b1
        second_test = torch.mm(torch.Tensor(x_list_test), w2) + b2
        third_test = torch.mm(torch.Tensor(x_list_test), w3) + b3
                
        y_hat_test = torch.softmax(torch.cat([first_test,second_test,third_test],axis = 1),dim=1)

        test_acc = torch.sum(torch.Tensor(label_list_test) == (torch.argmax(y_hat_test,axis =1).to(int)) ) / len(label_list_test)



        diff = (torch.linalg.norm(w1-w1_prev)+torch.linalg.norm(w2-w2_prev)+torch.linalg.norm(w3-w3_prev))/3
        print("loss = ",loss,"diff = ",diff,torch.sum(y == (torch.argmax(y_hat,axis =1).to(int)) ) / len(y),"test_acc = ",test_acc)

        # We will terminate when the difference in weights(averaged) is less than 10^(-4)
        if diff < 1e-4:
            break

    accuracy = torch.sum(y == (torch.argmax(y_hat,axis =1).to(int)) ) / len(y)
    print("accuracy obtained = ",accuracy)

    # dump it in the "w_b_file"
    with open(w_b_file, "wb") as file:
        pkl.dump([w1,b1,w2,b2,w3,b3], file)

    return w1,b1,w2,b2,w3,b3

# This function does the erm and returns the predictions
# beta_index is the index of the dominant feature
def erm(beta_index):

    # load the train data
    with open("train_2_class.pkl", "rb") as file:
        l = pkl.load(file)
    x_list, z_list, beta_list, label_list = l[0], l[1], l[2], l[3]


    w, b, l = train(np.array(x_list), np.array(label_list), bs=10000, epochs=100000, lr=0.1)
    print("weights = ",w)
    weights_vector = w


    # load the test data
    with open("test_2_class.pkl", "rb") as file:
        l = pkl.load(file)
    x_list, z_list, beta_list, label_list = l[0], l[1], l[2], l[3]


    prediction_result = predict(np.array(x_list),w,b)
    print(accuracy(np.array(label_list), y_hat=prediction_result))

    correct_beta_1 = 0
    total_beta_1 = 0
    correct_beta_0 = 0
    total_beta_0 = 0

    for i in range(len(x_list)):

        # beta_index is the index of the dominant feature whose accuracy we want when its beta is 1 and when it is zero
        if beta_list[i][beta_index]==1:
            correct_beta_1 += (label_list[i]==prediction_result[i])
            total_beta_1 += 1

        else:
            correct_beta_0 += (label_list[i]==prediction_result[i])
            total_beta_0 += 1        


    # beta_one_accuracy is the accuracy when the value of first beta is "1"
    beta_one_accuracy = correct_beta_1/total_beta_1
    beta_zero_accuracy = correct_beta_0/max(total_beta_0,1)
    total_accuracy = (correct_beta_1+correct_beta_0)/(total_beta_1+total_beta_0)

    print("beta one accuracy = ",beta_one_accuracy,"beta zero accuracy = ",beta_zero_accuracy)

    print("total accuracy = ",(correct_beta_1+correct_beta_0)/(total_beta_1+total_beta_0))

    return weights_vector, beta_one_accuracy*100, beta_zero_accuracy*100,total_accuracy*100

# This plots the accuracy on y axis and x axis is the mean of the dominant feature
def mu_plot(test_ratio,d,r,total_data,class_ratio,samples_per_instance):

    # arr is the mean "m" of the z1 that we will vary
    arr = [0.3-i*0.025 for i in range(13)]
    beta_one_accuracy_list = []
    beta_zero_accuracy_list = []
    minimum_accuracy_list = []

    for i in range(len(arr)):

        #generate the data and dump it
        generate_data_2_class(test_ratio,d,r,total_data,class_ratio,samples_per_instance,mean_step_size = arr[i])

        weights_vector, beta_one_accuracy, beta_zero_accuracy,total_accuracy = erm(0)
        beta_one_accuracy_list.append(beta_one_accuracy)
        beta_zero_accuracy_list.append(beta_zero_accuracy)

        minimum_accuracy = utils.overlap(arr[i],0.3,-arr[i],0.3)
        #NormalDist(mu=arr[i], sigma=0.3).overlap(NormalDist(mu=-1*arr[i], sigma=0.3))
        minimum_accuracy_list.append((minimum_accuracy)*100)



    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8))


    # Set position of bar on X axis
    br1 = np.arange(len(arr))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, beta_zero_accuracy_list, color ='r', width = barWidth,
            edgecolor ='grey', label ='beta_zero_accuracy')
    plt.bar(br2, beta_one_accuracy_list, color ='g', width = barWidth,
            edgecolor ='grey', label ='beta_one_accuracy')
    plt.bar(br3, minimum_accuracy_list, color ='b', width = barWidth,
            edgecolor ='grey', label ='minimum_accuracy')
    

    # Adding Xticks
    plt.xlabel('mean', fontweight ='bold', fontsize = 15)
    plt.ylabel('accuracy percentage', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(arr))],
            [round(i,3) for i in arr])

    plt.legend()
    plt.show()
    plt.savefig(f"mu_plot_{r}")

# Use this function if you want the total accuracy also to be displayed in the plot
# this will not generate data it will use the previously generated data
def mu_plot_with_total_accuracy(test_ratio,d,r,total_data,class_ratio,samples_per_instance):

    # arr is the mean "m" of the z1 that we will vary
    arr = [0.3-i*0.025 for i in range(13)]
    beta_one_accuracy_list = []
    beta_zero_accuracy_list = []
    minimum_accuracy_list = []
    total_accuracy_list = []

    for i in range(len(arr)):

        generate_data_2_class(test_ratio,d,r,total_data,class_ratio,samples_per_instance,mean_step_size = arr[i])

        weights_vector, beta_one_accuracy, beta_zero_accuracy,total_accuracy = erm(0)
        beta_one_accuracy_list.append(beta_one_accuracy)
        beta_zero_accuracy_list.append(beta_zero_accuracy)

        minimum_accuracy = utils.overlap(arr[i],0.3,-arr[i],0.3)
        #NormalDist(mu=arr[i], sigma=0.3).overlap(NormalDist(mu=-1*arr[i], sigma=0.3))
        minimum_accuracy_list.append((minimum_accuracy)*100)
        total_accuracy_list.append(total_accuracy)



    barWidth = 0.2
    fig = plt.subplots(figsize =(18, 12))


    # Set position of bar on X axis
    br1 = np.arange(len(arr))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    # Make the plot
    plt.bar(br1, beta_zero_accuracy_list, color ='r', width = barWidth,
            edgecolor ='grey', label ='beta_zero_accuracy')
    plt.bar(br2, beta_one_accuracy_list, color ='g', width = barWidth,
            edgecolor ='grey', label ='beta_one_accuracy')
    plt.bar(br3, minimum_accuracy_list, color ='b', width = barWidth,
            edgecolor ='grey', label ='minimum_accuracy')
    plt.bar(br4, total_accuracy_list, color ='y', width = barWidth,
            edgecolor ='grey', label ='total_accuracy')

    # Adding Xticks
    plt.xlabel('mean', fontweight ='bold', fontsize = 15)
    plt.ylabel('accuracy percentage', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(arr))],
            [round(i,3) for i in arr])

    plt.legend()
    plt.show()



if __name__ == "__main__":

    # This is to generate the accuracy plot for the 2 class dataset
    # mu_plot(test_ratio = 0.4,d=10,r=5,total_data = 10000,class_ratio = 0.5,samples_per_instance = 3)

    train_data = 6000
    # Note that test_data and val_data will be divided by sample_per_instance 
    test_data = 4000
    val_data = 4000
    samples_per_instance = 10
    class_ratio_first = 1/3
    class_ratio_second = 1/3
    class_ratio_third = 1/3
    mean_step_size = 0.3
    d=10
    r=5

    mu_first_list = [0.1,0.3,0.1,0.1,-0.3,0.1,0.1,-0.3,0.1,0.1]
    mu_second_list = [0.1,-0.3,0.1,0.1,0.3,0.1,0.1,-0.3,0.1,0.1]
    mu_third_list = [0.1,-0.3,0.1,0.1,-0.3,0.1,0.1,0.3,0.1,0.1]

    sigma_first_list = [0.05   for i in range(d)]
    sigma_second_list = [0.05   for i in range(d)]
    sigma_third_list = [0.05   for i in range(d)]

    train_path = "train_3_class.pkl"
    test_path = "test_3_class.pkl"
    val_path = "val_3_class.pkl"
    process_train_path = "processed_train_3_class.pkl"
    R_dump_path = "final_R_3_class_3000.pkl"
    w_b_file = "w_b_file_3_class.pkl"

    # This will generate the datasets
    generate_data_3_class_dataset(d,r,train_data,class_ratio_first,class_ratio_second,class_ratio_third,samples_per_instance,mean_step_size,mu_first_list,mu_second_list,mu_third_list,sigma_first_list,sigma_second_list,sigma_third_list,data_type="train",dump_path=train_path)
    generate_data_3_class_dataset(d,r,test_data,class_ratio_first,class_ratio_second,class_ratio_third,samples_per_instance,mean_step_size,mu_first_list,mu_second_list,mu_third_list,sigma_first_list,sigma_second_list,sigma_third_list,data_type="test",dump_path=test_path)
    generate_data_3_class_dataset(d,r,val_data,class_ratio_first,class_ratio_second,class_ratio_third,samples_per_instance,mean_step_size,mu_first_list,mu_second_list,mu_third_list,sigma_first_list,sigma_second_list,sigma_third_list,data_type="val",dump_path=val_path)

    # This will process the train data 
    process_train(file_name=train_path,dump_path=process_train_path)

    # This will do the gradient descent
    grad_descent_torch(file_name=process_train_path,test_file_name=test_path,w_b_file=w_b_file)

    # We will load the dumped weights
    with open(w_b_file, 'rb') as file:
        r = pkl.load(file)

    w1 = r[0]
    b1 = r[1]
    w2 = r[2]
    b2 = r[3]
    w3 = r[4]
    b3 = r[5]        

    # We will use this function to select elements into R and dump it in "R_dump_path" and also update the weights in the process
    recourse(w1,b1,w2,b2,w3,b3,process_train_path=process_train_path,R_dump_path=R_dump_path,test_file_name=test_path,num_R=3000,num_randomly_sample_indices=100,num_grad_descent_iterations=10)
    
    # Load the R array and the updated weights
    with open(R_dump_path, 'rb') as file:
        r = pkl.load(file)

    R = r[0]    
    w1 = r[1]
    b1 = r[2]
    w2 = r[3]
    b2 = r[4]
    w3 = r[5]
    b3 = r[6]

    logger_file = config["logger_file"]

    logger = utils.init_logger(f"logs/{logger_file}.log", file_mode="a")
    logger.info(utils.pretty_print(config))

    # Declare phi model
    phi_layers = config["phi"]["layers"]
    dropouts = config["phi"]["dropouts"]
    phi = PhiModel(phi_layers, dropouts=dropouts, logger=logger)

    phi.to(constants.device)


    # Declare psi model
    psi_layers = config["psi"]["layers"]
    dropouts = config["psi"]["dropouts"]
    psi = PsiModel(psi_layers, dropouts=dropouts, logger=logger)

    psi.to(constants.device)

    load_path = "test_3_class.pkl"
    output_R_path = "output_R_test_3_class.pkl"

    # We load the psi model 
    psi.load_psi()
    psi.freeze()
    psi.eval()
    # The below function will predict the R values for the test data
    psi.output_R(load_path = load_path,output_R_path = output_R_path)

    # We load the phi model
    phi.load_phi()
    phi.freeze()
    phi.eval()
    # The below function will take the predicted R values and then predict the beat values for these
    phi.output_beta(load_path = load_path,output_R_path = output_R_path,w1 = w1,b1 = b1,w2 = w2,b2 = b2,w3 = w3,b3 = b3,num_beta_to_unmask = 1)
