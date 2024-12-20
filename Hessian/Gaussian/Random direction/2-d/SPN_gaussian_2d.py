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
from matplotlib import rcParams
import pandas as pd
import plotly
import plotly.graph_objects as go
import plotly.express as px

rcParams['font.weight'] = 'bold'

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)
device = torch.device(device)
batchsize = 1

mu1,mu2,mu3 = 5,1.5,8
sigma1,sigma2,sigma3 = 0.2,0.5,0.3

Latents = [5,20,50]
depth = 0
num_repetitions = 1
instance = 10
Points = np.array([50,1000,3000])
#print(mu,sigma)

 
# Setting mean of the distributino 
# to be at (0,0)
mean1 = np.array([-5, -5])
mean2 = np.array([5, 5])
mean3 = np.array([-5, 5])
# Storing density function values for 
# further analysis
pdf_list = []
 
# Iterating over different covariance values
   
# Initializing the covariance matrix
cov1 = np.array([[1, 0], [0, 1]])
cov2 = np.array([[1, 0], [0, 1]])
cov3 = np.array([[1, 0], [0, 1]])        
# Generating a Gaussian bivariate distribution
# with given mean and covariance matrix
distr1 = multivariate_normal(cov = cov1, mean = mean1)
distr2 = multivariate_normal(cov = cov2, mean = mean2)
distr3 = multivariate_normal(cov = cov3, mean = mean3) 


data1 = np.random.multivariate_normal(mean1, cov1, 10000)
data2 = np.random.multivariate_normal(mean2, cov2, 10000)
data3 = np.random.multivariate_normal(mean3, cov3, 10000)

x_orig = []
for i in range(len(data1)):
    x_orig.append(data1[i])
for i in range(len(data2)):
    x_orig.append(data2[i])
for i in range(len(data3)):
    x_orig.append(data3[i])
    
data1 = np.random.multivariate_normal(mean1, cov1, 100)
data2 = np.random.multivariate_normal(mean2, cov2, 100)
data3 = np.random.multivariate_normal(mean3, cov3, 100)

x_test = []
for i in range(len(data1)):
    x_test.append(data1[i])
for i in range(len(data2)):
    x_test.append(data2[i])
for i in range(len(data3)):
    x_test.append(data3[i])
    
x_orig = np.array(x_orig)   
X_orig = x_orig.reshape([len(x_orig),2]) 

test_x_orig = np.array(x_test)   
test_x_orig = np.sort(test_x_orig)
test_x = test_x_orig.reshape([len(test_x_orig),2])

  
# Generating a meshgrid complacent with
# the 3-sigma boundary

qwert = 1     
x1 = np.linspace(-10,10,100)
y1 = np.linspace(-10,10,100)
X, Y = np.meshgrid(x1,y1)
     
# Generating the density function
# for each point in the meshgrid
pdf1 = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pdf1[i,j] = distr1.pdf([X[i,j], Y[i,j]])

pdf2 = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pdf2[i,j] = distr2.pdf([X[i,j], Y[i,j]])
        
pdf3 = np.zeros(X.shape)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        pdf3[i,j] = distr3.pdf([X[i,j], Y[i,j]])        

PDF = pdf1/2+pdf2/4+pdf3/4  


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, PDF, cmap='viridis')
plt.savefig('2D_Gaussian')

for num_latents in Latents:

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

            train_x_random = x_orig
            np.random.shuffle(train_x_random)
            train_x = train_x_random[:points]
            train_x = train_x.reshape([len(train_x),2])
 
    
            #print(train_x)
            # to torch
            train_x = torch.tensor(train_x,dtype=torch.float64)

            test_x = torch.tensor(test_x,dtype=torch.float64)
 
            train_N= train_x.shape
  
            test_N = test_x.shape[0]


            ns = juice.structures.RAT_SPN(num_vars = len(train_x[0]), 
                    num_latents = num_latents,
                    depth = depth,
                    num_repetitions = num_repetitions,
                    input_node_type = Gaussian
                )
    
            pc = juice.compile(ns)
  
            pc.to(device)
            
            if qwert ==2:
                plt.figure()
                juice_vis.plot_pc(ns, node_id = True, node_num_label = True)
                plt.savefig('SPN_Gaussian_'+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
                qwert =2         
            optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1, method = "EM")
            scheduler = juice.optim.CircuitScheduler(
                    optimizer, 
                    method = "multi_linear", 
                    lrs = [0.9, 0.1, 0.05], 
                    milestone_steps = [0, len(train_x) * 100, len(train_x) * 350]
                )

            #print('before training')
            PC_state = pc.state_dict()
            params = PC_state['params']
            input_params =  PC_state['input_layer_group.layer_0.params']
            #print('params = ',params)
            #print('Input_params = ', input_params)
            lls = pc(train_x.to(pc.device))
            #print('Train LL = ',lls.mean().detach().cpu().numpy().item())
    

            Train_ll = []
            Test_ll =[]
            start_time = time.time()
            for epoch in range(1, 1550+1):
                t0 = time.time()
                train_ll = []
                TRAIN_ll =[]
         
                x = train_x.to(device)
   
                # Similar to PyTorch optimizers zeroling out the gradients, we zero out the parameter flows
                optimizer.zero_grad()
                # Forward pass
                lls = pc(x)

                # Backward pass
                lls.mean().backward()
                TRAIN_ll = lls.detach().cpu().numpy()
                train_ll = lls.mean().detach().cpu().numpy().item()

                # Perform a mini-batch EM step  
                optimizer.step()
                scheduler.step()
                Train_ll.append(train_ll)
                t1 = time.time()
 

            exp_train =[]
            for x in TRAIN_ll:
                exp_train.append(np.exp(x)) 

   
            #print('after training')
            PC_state = pc.state_dict()
            params = PC_state['params']
            input_params =  PC_state['input_layer_group.layer_0.params']
            print('params = ',params)
            #print('params = ',sum(params))
            print('Input_params = ', input_params)
            lls = pc(train_x.to(pc.device))
            training_ll = lls.mean().detach().cpu().numpy().item()
            #print('Train LL = ',training_ll)
            Train_ll_points.append(training_ll)
    
            llst = pc(test_x.to(pc.device))
            testing_ll = llst.mean().detach().cpu().numpy().item()
            #print('Test LL = ',testing_ll)
            
  
            TEST_ll = llst.detach().cpu().numpy()
            Test_ll_points.append(testing_ll)
            #diff.append(abs(testing_ll-training_ll))
            exp_test =[]
            for x in TEST_ll:
                exp_test.append(np.exp(x)) 
            print(len(test_x[:,0]))
            print(len(exp_test))
            fig = plt.figure(figsize=(12, 12))
            ax = fig.add_subplot(projection='3d')
            ax.plot_surface(X, Y, PDF, alpha=0.5,cmap='viridis')
            ax.scatter3D(test_x[:,0], test_x[:,1],exp_test , color = "red")
            plt.savefig('2D_Gaussian_ModeCollapse'+str(depth)+'_'+str(num_latents)+'_'+str(points))




