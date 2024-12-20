import pyjuice as juice
import pyjuice.visualize as juice_vis
from pyjuice.nodes.distributions import *
import torch
import torchvision
import time
import pyjuice.nodes.distributions as dists
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib
import pandas as pd

matplotlib.rcParams.update({'font.size': 35})

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)
device = torch.device(device)
batchsize = 1

train_dataset = torchvision.datasets.MNIST(root = "../data", train = True, download = True)
valid_dataset = torchvision.datasets.MNIST(root = "../data", train = False, download = True)

train_data = train_dataset.data.reshape(60000, 28*28)
valid_data = valid_dataset.data.reshape(10000, 28*28)

train_loader = DataLoader(
    dataset = TensorDataset(train_data),
    batch_size = 512,
    shuffle = True,
    drop_last = True
)
valid_loader = DataLoader(
    dataset = TensorDataset(valid_data),
    batch_size = 512,
    shuffle = False,
    drop_last = True
)


trainData = []
for i in range(len(train_dataset.data)):
    a = train_dataset.data[i]
    c = []
    for j in range(len(a)):
        a2 = []
        #if j%2 ==0:
        for k in range(len(a[0])):
            #if k%2==0:
            if a[j][k]>150:
                a2.append(1)
            else:
                a2.append(0)
        c.append(a2) 
    trainData.append(c)        

validData = []
for i in range(len(valid_dataset.data)):
    a = valid_dataset.data[i]
    c = []
    for j in range(len(a)):
        a2 = []
        #if j%2 ==0:
        for k in range(len(a[0])):
            #if k%2==0:
            if a[j][k]>150:
                a2.append(1)
            else:
                a2.append(0)
        c.append(a2) 
    validData.append(c)  

train_data = torch.tensor(np.array(trainData).reshape(60000, 28*28))
valid_data = torch.tensor(np.array(validData).reshape(10000, 28*28))

train_loader = DataLoader(
    dataset = TensorDataset(train_data),
    batch_size = 512,
    shuffle = True,
    drop_last = True
)
valid_loader = DataLoader(
    dataset = TensorDataset(valid_data),
    batch_size = 512,
    shuffle = False,
    drop_last = True
)


LATENTS = [[16,32,64]]
Points = [0]
Depth = [5]
num_repetitions = 1
instance = 10
DIM = [0]
for dim in DIM:


    Points = np.array([5000])

    for depth in Depth:
        qwert = 1
        for Latents in LATENTS:

            Flat =[]
            norm_flat = []
            diff_ = []
            for num_latents in Latents:
                print(dim,depth,num_latents)
                train_average = []
                test_average = []


                #X_orig = np.random.normal(mu1,sigma1,[1000,1])
                TRAIN_like = np.zeros(150)
                TEST_like = np.zeros(150)
                for trial in range(1):
                    Train_ll_points =[]
                    Test_ll_points =[]
                    Flatness =[]   
                    norm_flatness = []
                    for points in Points:

                       
                        train_x = train_data
 
    
                        #print(train_x)
                        # to torch
                        train_x = torch.tensor(train_x,dtype=torch.float64)

                        test_x = torch.tensor(valid_data,dtype=torch.float64)
 
                        train_N= train_x.shape
  
                        test_N = test_x.shape[0]
                        
                        ns = juice.structures.RAT_SPN(num_vars = len(train_x[0]), 
                                num_latents = num_latents,
                                depth = depth,
                                num_repetitions = num_repetitions
                            )
                        pc = juice.compile(ns)
  
                        pc.to(device)
                        if qwert ==1:
                            plt.figure()
                            juice_vis.plot_pc(ns, node_id = True, node_num_label = True)
                            #plt.title('SPN structure'+'_dim = '+ str(dim)+'_latents = '+str(num_latents)+'_depth = '+ str(depth))
                            plt.savefig('SPN_Gaussian_'+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
                            qwert =2         
                        optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1, method = "EM")
                        scheduler = juice.optim.CircuitScheduler(
                                optimizer, 
                                method = "multi_linear", 
                                lrs = [0.9, 0.1, 0.05], 
                                milestone_steps = [0, len(train_x) * 100, len(train_x) * 350]
                            )

                        print('before training')
                        PC_state = pc.state_dict()
                        params = PC_state['params']
                        input_params =  PC_state['input_layer_group.layer_0.params']
                        #print('params = ',params)
                        #print('Input_params = ', input_params)
                        for batch in train_loader:
                            x = batch[0].to(device)

                            lls = pc(x, record_cudagraph = True)
                            lls.mean().backward()
                            break

                        training_ll = lls.mean().detach().cpu().numpy().item()                            
                        print('Train LL = ',training_ll)
    

                        Train_ll = []
                        Test_ll =[]
                        start_time = time.time()
                        for epoch in range(1, 150+1):
                            t0 = time.time()
                            train_ll = 0
                            TRAIN_ll =[]
                            for batch in train_loader:
                                x = batch[0].to(device)

                                # Similar to PyTorch optimizers zeroling out the gradients, we zero out the parameter flows
                                optimizer.zero_grad()

                                # Forward pass
                                lls = pc(x)

                                # Backward pass
                                lls.mean().backward()

                                train_ll += lls.mean().detach().cpu().numpy().item()

                                # Perform a mini-batch EM step
                                optimizer.step()
                                scheduler.step()

                            train_ll /= len(train_loader)
                            t1 = time.time()
                            test_ll = 0.0
                            for batch in valid_loader:
                                x = batch[0].to(pc.device)
                                lls = pc(x)
                                test_ll += lls.mean().detach().cpu().numpy().item()
    
                            test_ll /= len(valid_loader)
                            t2 = time.time()
                            print(train_ll)
                            print(test_ll)
                            Train_ll.append(train_ll)
                            Test_ll.append(test_ll)
                            t1 = time.time()
                    TRAIN_like = TRAIN_like + np.array(Train_ll)
                    TEST_like = TEST_like + np.array(Test_ll)
                # Figure size 
                plt.figure(figsize=(25,20))
                plt.xlabel('Epoch')
                plt.ylabel('likelihood')
                # Plotting
                #sort = np.argsort(avg_flat)
                plt.plot(TRAIN_like/1, marker = 'o', label='Train Likelihood')
                plt.plot(TEST_like/1, marker = 'o', label=' Test Likelihood')
                plt.title('Difference in likelihood' + '_'+str(num_latents)+ '_'+str(depth))
                plt.legend()
                plt.savefig(' likelihood_28_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
                plt.close()     
     
