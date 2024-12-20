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

mu1,mu2,mu3 = 5,1.5,8
sigma1,sigma2,sigma3 = 0.2,0.5,0.3

LATENTS = [[20,30,50]]

Depth = [0]
num_repetitions = 1
instance = 10

#print(mu,sigma)

def generate_data(num_samples=100):
    # Parameters for first Gaussian blob
    mean1 = [2.0, 3.0]
    cov1 = [[0.4, 0.3], [0.3, 0.4]]

    # Parameters for second Gaussian blob
    mean2 = [-1.0, -2.0]
    cov2 = [[0.4, 0.2], [0.2, 0.4]]

    # Parameters for third Gaussian blob
    mean3 = [4.0, -1.0]
    cov3 = [[0.3, 0.2], [0.2, 0.3]]

    # Parameters for fourth Gaussian blob
    mean4 = [-3.0, 2.0]
    cov4 = [[1, 0], [0, 1]]

    # Generate data points
    data1 = np.random.multivariate_normal(mean1, cov1, num_samples // 4)
    data2 = np.random.multivariate_normal(mean2, cov2, num_samples // 4)
    data3 = np.random.multivariate_normal(mean3, cov3, num_samples // 4)
    data4 = np.random.multivariate_normal(mean4, cov4, num_samples // 4)
    data = np.vstack([data1, data2, data3, data4])

    return torch.tensor(data, dtype=torch.float32)

# Setting mean of the distributino 
# to be at (0,0)
mean1 = np.array([5.0, -5.0])
mean2 = np.array([5, 5])
mean3 = np.array([-5, 5])
 
# Storing density function values for 
# further analysis
pdf_list = []
 
# Iterating over different covariance values
   
# Initializing the covariance matrix
cov1 = np.array([[0.4, 0.3], [0.3, 0.4]])
cov2 = np.array([[0.6, -0.3], [-0.3, 0.9]])
cov3 = np.array([[0.5, 0.2], [0.2, 0.8]])     
# Generating a Gaussian bivariate distribution
# with given mean and covariance matrix
distr1 = multivariate_normal(cov = cov1, mean = mean1)
distr2 = multivariate_normal(cov = cov2, mean = mean2)
distr3 = multivariate_normal(cov = cov3, mean = mean3) 

data1 = distr1.rvs(size = 100000)
data2 = distr2.rvs(size = 50000)
data3 = distr3.rvs(size = 50000)  

x_orig = []
for i in range(len(data1)):
    x_orig.append(data1[i])
for i in range(len(data2)):
    x_orig.append(data2[i])
for i in range(len(data3)):
    x_orig.append(data3[i])
        
data1 = distr1.rvs(size = 100)
data2 = distr2.rvs(size = 50)
data3 = distr3.rvs(size = 50)

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

Points = np.array([50,100,200,300,500,1000,1250,1500,2000])

for depth in Depth:
    qwert = 1
    for Latents in LATENTS:

        Flat =[]
        norm_flat = []
        diff_ = []
        for num_latents in Latents:

            train_average = []
            test_average = []


            #X_orig = np.random.normal(mu1,sigma1,[1000,1])
            avg_diff = np.zeros(len(Points))
            avg_flat = np.zeros(len(Points))
            avg_normflat = np.zeros(len(Points))
            for trial in range(100):
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
                    if qwert ==1:
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

                    print('before training')
                    PC_state = pc.state_dict()
                    params = PC_state['params']
                    input_params =  PC_state['input_layer_group.layer_0.params']
                    #hess = torch.autograd.functional.hessian(pc, train_x.to(device)[6].reshape([1,2]), create_graph=False, strict=False, vectorize=False, outer_jacobian_strategy='reverse-mode')
                    #print()
                    #print(hess[0])
                    #print()
                    #print('params = ',params)
                    print('Input_params = ', input_params)
                    lls = pc(train_x.to(pc.device))
                    #print('Train LL = ',lls.mean().detach().cpu().numpy().item())
    

                    Train_ll = []
                    Test_ll =[]
                    start_time = time.time()
                    for epoch in range(1, 150+1):
                        t0 = time.time()
                        train_ll = []
                        TRAIN_ll =[]
         
                        x = train_x.to(device)
   
                        # Similar to PyTorch optimizers zeroling out the gradients, we zero out the parameter flows
                        optimizer.zero_grad()
                        # Forward pass
                        PC_state1 = pc.state_dict()
                        PC_state1['input_layer_group.layer_0.params'] = input_params
                        pc.load_state_dict(PC_state1)                        
                        lls = pc(x)

                        # Backward pass
                        lls.mean().backward()
                        TRAIN_ll = lls.detach().cpu().numpy()
                        train_ll = lls.mean().detach().cpu().numpy().item()

                        # Perform a mini-batch EM step  
                        optimizer.step()
                        scheduler.step()
                        PC_state1 = pc.state_dict()
                        PC_state1['input_layer_group.layer_0.params'] = input_params
                        pc.load_state_dict(PC_state1) 
                        Train_ll.append(train_ll)
                        t1 = time.time()
 
 
                    exp_train =[]
                    for x in TRAIN_ll:
                        exp_train.append(np.exp(x)) 

   
                    print('after training')
                    PC_state = pc.state_dict()
                    params = PC_state['params']
                    input_params =  PC_state['input_layer_group.layer_0.params']
                    #print('params = ',params)
                    #print('params = ',sum(params))
                    print('Input_params = ', input_params)
                    lls = pc(train_x.to(pc.device))
                    training_ll = lls.mean().detach().cpu().numpy().item()
                    print(train_x.to(device)[0])
                    print(train_x.to(device)[0].reshape([1,2]).dim())
                    #hess = torch.autograd.functional.hessian(pc, train_x.to(device)[6].reshape([1,2]), create_graph=False, strict=False, vectorize=False, outer_jacobian_strategy='reverse-mode')
                    #print()
                    #print(hess)
                    #print()
                    print('Train LL = ',training_ll)
                    Train_ll_points.append(training_ll)
    
                    llst = pc(test_x.to(pc.device))
                    testing_ll = llst.mean().detach().cpu().numpy().item()
                    print('Test LL = ',testing_ll)
            
  
                    TEST_ll = llst.detach().cpu().numpy()
                    Test_ll_points.append(testing_ll)
                    #diff.append(abs(testing_ll-training_ll))
                    exp_test =[]
                    for x in TEST_ll:
                        exp_test.append(np.exp(x)) 
 
                    pc_toy = juice.compile(ns)
                    pc_toy.to(device)
                    PC_state = pc.state_dict()
                    #print(PC_state)
                    params = PC_state['params'].cpu()
                    direction = np.zeros(len(params))
              
                    ii = np.sum(np.array(params)==0)
                    print(ii)
                    for i in range(ii,int(len(params))-1,2):
                        m = min(params[i],params[i+1])
                        d = np.random.normal(m/2,m/6)
                        direction[i] = d
                        direction[i+1] = -d    
                    #lls = pc_toy(train_x.to(pc.device))
                    #print('Test LL new = ',lls.mean().detach().cpu().numpy().item())
                    PC_train_ll = []
                    A = np.linspace(-0.2,0.2,100)
                    direction = torch.tensor(np.array(direction)).to(pc.device)
                    #print('Direction = ',direction)
                    #print('direction = ',direction)
                    for a in A:
                        PC_state['params'] = params.to(pc.device)+a*direction
                        pc_toy.load_state_dict(PC_state)
                  
                        x = train_x.to(pc.device)
                        lls = pc_toy(x)
                        train_ll = lls.mean().detach().cpu().numpy().item()
                                   
                        PC_train_ll.append(train_ll) 
            
                    f1 = abs(PC_train_ll[0]-max(PC_train_ll))
                    f2 = abs(PC_train_ll[-1]-max(PC_train_ll))
                    Flatness.append(max(f1,f2))

                    norm_direction = np.zeros(len(params))
                              
                    for i in range(ii,int(len(params))-1,2):
                 
                        m = min(params[i],params[i+1])
                        d = np.random.normal(m/2,m/6)
                        norm_direction[i] = d
                        norm_direction[i+1] = -d 
                    norm1 = np.linalg.norm(norm_direction[ii:])
                    norm2 = np.linalg.norm(params[ii:]) 
                    norm_direction = (np.array(norm_direction)/norm1)*norm2
                              
                    PC_train_ll_norm = []
                    norm_direction = torch.tensor(norm_direction).to(pc.device)
                    #print(norm_direction)
                    #print('direction = ',norm_direction)
                    A = np.linspace(-0.05,0.05,100)
                    for a in A:
                        PC_state['params'] = params.to(pc.device)+a*norm_direction
                        pc_toy.load_state_dict(PC_state)
                    
                        x = train_x.to(pc.device)
                        lls = pc_toy(x)
                        train_ll = lls.mean().detach().cpu().numpy().item()
                                
                        PC_train_ll_norm.append(train_ll)
                    #print(PC_train_ll_norm)
                    f1 = abs(PC_train_ll_norm[0]-max(PC_train_ll_norm))
                    f2 = abs(PC_train_ll_norm[-1]-max(PC_train_ll_norm))
                    norm_flatness.append(max(f1,f2))            

     
                avg_flat = avg_flat+np.array(Flatness)
                avg_normflat = avg_normflat+np.array(norm_flatness)
                avg_diff = avg_diff+np.array(abs(np.array(Train_ll_points)-np.array(Test_ll_points)))


           # Figure size 
            plt.figure(figsize=(15,12))
            plt.xlabel('Peakness')
            plt.ylabel('likelihood')
            # Plotting
            plt.scatter(avg_flat/100,avg_diff/100, label=' difference Likelihood')
            plt.title('Difference in likelihood vs flat' + '_'+str(num_latents)+'_'+ str(depth))
            plt.legend(loc = "upper left")
            plt.savefig('avg Train,Test vs flat_geom_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
            plt.close()     

            # Figure size
            plt.figure(figsize=(15,12))
            plt.xlabel('Peakness')
            plt.ylabel('likelihood')
            # Plotting
            plt.scatter(avg_normflat/100,avg_diff/100 , label=' difference Likelihood')
            plt.title('Difference in likelihood vs norm flat' + '_'+str(num_latents)+'_'+ str(depth))
            plt.legend(loc = "upper left")
            plt.savefig('avg Train,Test vs flat_geom_norm1'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
            plt.close()     

            # Figure size
            plt.figure(figsize=(15,12))
            plt.xlabel('Points')
            plt.ylabel('likelihood')
            # Plotting
            plt.scatter(Points,avg_diff/100 , label=' difference Likelihood')
            plt.title('Difference in likelihood vs point' + '_'+str(num_latents)+'_'+ str(depth))
            plt.legend(loc = "upper right")
            plt.savefig('Difference in likelihood vs Points_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
            plt.close()     

            Flat.append(avg_flat/100)
            norm_flat.append(avg_normflat/100)
            diff_.append(avg_diff/100)

  
        # Figure size 
        plt.figure(figsize=(15,12)) 
        plt.xlabel('Peakness')
        plt.ylabel('likelihood')
        # Plotting
        plt.scatter(Flat[0],diff_[0], label= 'Latent size = '+str(Latents[0]))
        plt.scatter(Flat[1],diff_[1], label= 'Latent size = '+str(Latents[1]))
        plt.scatter(Flat[2],diff_[2], label= 'Latent size = '+str(Latents[2]))
        plt.title('Difference in likelihood vs flat (across different latent size)'+'_'+ str(depth))
        plt.legend(loc="upper left")
        plt.savefig('avg Train,Test vs flat_geom1_'+str(Latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
        plt.close()     

        # Figure size
        plt.figure(figsize=(15,12))
        plt.xlabel('Peakness')
        plt.ylabel('likelihood')
        # Plotting
        plt.scatter(norm_flat[0],diff_[0], label= 'Latent size = '+str(Latents[0]))
        plt.scatter(norm_flat[1],diff_[1], label= 'Latent size = '+str(Latents[1]))
        plt.scatter(norm_flat[2],diff_[2], label= 'Latent size = '+str(Latents[2]))
        plt.title('Difference in likelihood vs norm flat (across different latent size)'+'_'+ str(depth))
        plt.legend(loc="upper left")
        plt.savefig('avg Train,Test vs flat_geom_norm1_'+str(Latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
        plt.close()     
 
        # Figure size
        plt.figure(figsize=(15,12))
        plt.xlabel('Points')
        plt.ylabel('likelihood')
        # Plotting
        plt.scatter(Points,diff_[0], label= 'Latent size = '+str(Latents[0]))
        plt.scatter(Points,diff_[1], label= 'Latent size = '+str(Latents[1]))
        plt.scatter(Points,diff_[2], label= 'Latent size = '+str(Latents[2]))
        plt.title('Difference in likelihood vs point (across different latent size)'+'_'+ str(depth))
        plt.legend(loc = "upper right")
        plt.savefig('Difference in likelihood vs Points_'+str(Latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
        plt.close()      
