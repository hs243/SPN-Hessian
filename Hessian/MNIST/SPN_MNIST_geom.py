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

matplotlib.rcParams.update({'font.size': 22})

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

Latents = [20]
Points = [0]
Depth = [5]
num_repetitions = 1
instance = 10

for depth in Depth:
    qwert = 1
    
    for num_latents in Latents:
        print(depth,num_latents)
        train_average = []
        test_average = []


        #X_orig = np.random.normal(mu1,sigma1,[1000,1])
        avg_diff = np.zeros(len(Points))
        avg_flat = np.zeros(len(Points))
        avg_normflat = np.zeros(len(Points))
        for trial in range(1):
            Train_ll_points =[]
            Test_ll_points =[]
            Flatness =[]   
            norm_flatness = []
            for points in Points:

 
                #print(train_x)
                # to torch
                train_x = torch.tensor(train_data,dtype=torch.float64)
                print(train_x.shape)
                ns = juice.structures.RAT_SPN(num_vars = len(train_x[0]), 
                        num_latents = num_latents,
                        depth = depth,
                        num_repetitions = num_repetitions
                    )
    
                pc = juice.compile(ns)
                pc.to(device)
  
                if qwert ==0:
                    plt.figure()
                    juice_vis.plot_pc(ns, node_id = True, node_num_label = True)
                    #plt.title('SPN structure'+'_dim = '+ str(dim)+'_latents = '+str(num_latents)+'_depth = '+ str(depth))
                    plt.savefig('SPN_Gaussian_'+'_'+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
                    qwert =2         
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

