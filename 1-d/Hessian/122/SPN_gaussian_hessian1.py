import pyjuice as juice
import pyjuice.visualize as juice_vis
from pyjuice.nodes.distributions import *
from pyjuice.nodes.nodes import CircuitNodes
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
import math

matplotlib.rcParams.update({'font.size': 40})

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)
device = torch.device(device)
batchsize = 1

mu1,mu2,mu3 = 5,3.5,7
sigma1,sigma2,sigma3 = 0.2,0.5,0.3

LATENTS = [[10,20,50]]

Depth = [0]
num_repetitions = 1
trial = 1
iterr = 0
DIM = [2]
#print(mu,sigma)

def Gauss(mu,sig,x):
    f = (1/(sig*math.sqrt(2*math.pi)))*math.exp(-0.5*((x-mu)/sig)**2)
    return f
"""
XX = np.linspace(-2.0, 2.0, num=100)
normal = []
for i in range(len(XX)):
    normal.append(Gauss(0,0.5,XX[i]))   
plt.plot(XX,normal)
plt.show()    
"""
def Derivative(pc):
    PC_state = pc.state_dict()
    params = PC_state['params']
    Params = params[params.nonzero()].squeeze(1).detach().cpu().numpy()
    Flows = pc.param_flows.detach().cpu().numpy()
    Derivatives = []
    for ik in range(len(Flows)):
        Derivatives.append(Flows[ik]/Params[ik])
    return Derivatives    

def HESSIAN(pc):
    Derivatives = Derivative(pc)
    #print(Derivatives)
    Hessian = []
    for ik in range(len(Derivatives)):
        row = []
        for il in range(len(Derivatives)):
            row.append(-1*Derivatives[ik]*Derivatives[il])
        Hessian.append(row) 
    return np.matrix(Hessian)

def SGM(Hessian):    
    #print('Derivative computed using Flows = ',Derivatives)                                    
    eigenvalues, eigenvectors = np.linalg.eig(Hessian)
    return eigenvalues


for dim in DIM:

    # Setting mean of the distributino 
    # to be at (0,0)
    mean1 = np.array([mu1]*dim)
    mean2 = np.array([mu2]*dim) 
    mean3 = np.array([mu3]*dim)

 
    # Iterating over different covariance values
   
    # Initializing the covariance matrix
    cov1 = np.eye(dim)
    cov2 = np.eye(dim)
    cov3 = np.eye(dim)     
    # Generating a Gaussian bivariate distribution
    # with given mean and covariance matrix
    distr1 = multivariate_normal(cov = cov1, mean = mean1)
    distr2 = multivariate_normal(cov = cov2, mean = mean2)
    distr3 = multivariate_normal(cov = cov3, mean = mean3) 

    data1 = distr1.rvs(size = 1000000)
    data2 = distr2.rvs(size = 500000)
    data3 = distr3.rvs(size = 500000)  

    x_orig = []
    for i in range(len(data1)):
        x_orig.append(data1[i])
    for i in range(len(data2)):
        x_orig.append(data2[i])
    for i in range(len(data3)):
        x_orig.append(data3[i])
        
    data1 = distr1.rvs(size = 1000)
    data2 = distr2.rvs(size = 500)
    data3 = distr3.rvs(size = 500)

    x_test = []
    for i in range(len(data1)):
        x_test.append(data1[i])
    for i in range(len(data2)):
        x_test.append(data2[i])
    for i in range(len(data3)):
        x_test.append(data3[i])
    
    x_orig = np.array(x_orig)   
    X_orig = x_orig.reshape([len(x_orig),dim]) 

    test_x_orig = np.array(x_test)   
    test_x_orig = np.sort(test_x_orig)
    test_x = test_x_orig.reshape([len(test_x_orig),dim])



    Points = np.array([30,750,3000,6000,10000])

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

                EigVal = np.zeros(num_latents)
                
                #X_orig = np.random.normal(mu1,sigma1,[1000,1])
                avg_diff = np.zeros(len(Points))
                avg_flat = np.zeros(len(Points))
                avg_normflat = np.zeros(len(Points))
                for tr in range(trial):
                    Train_ll_points =[]
                    Test_ll_points =[]
                    Flatness =[]   
                    norm_flatness = []
                    for points in Points:

                        train_x_random = x_orig
                        np.random.shuffle(train_x_random)
                        train_x = train_x_random[:points]
                        train_x = train_x.reshape([len(train_x),dim])
 
    
                        #print(train_x)
                        # to torch
                        train_x = torch.tensor(train_x,dtype=torch.float64, requires_grad=True)
                        if iterr == 0:
                            test_x = torch.tensor(test_x,dtype=torch.float64, requires_grad=True)
                        else:
                            test_x = test_x.clone().detach().requires_grad_(True)
                        iterr = 1  
                        train_N= train_x.shape
  
                        test_N = test_x.shape[0]


                        ns = juice.structures.RAT_SPN(num_vars = len(train_x[0]), 
                                num_latents = num_latents,
                                depth = depth,
                                num_repetitions = num_repetitions,
                                input_node_type = Gaussian
                            )
                        
                        pc = juice.compile(ns)
                        
                        PC_state = pc.state_dict()
                        #print(pc) 
                        #pc.requires_grad_(True)
                        pc.load_state_dict(PC_state)
                        print('before training')
                       
                        pc.to(device)
                        """
                        if qwert ==1:
                            plt.figure()
                            juice_vis.plot_pc(ns, node_id = True, node_num_label = True)
                            #plt.title('SPN structure'+'_dim = '+ str(dim)+'_latents = '+str(num_latents)+'_depth = '+ str(depth))
                            plt.savefig('SPN_Gaussian_'+'_'+ str(dim)+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
                            qwert =2 
                        """            
                        optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1, method = "EM")
                        scheduler = juice.optim.CircuitScheduler(
                                optimizer, 
                                method = "multi_linear", 
                                lrs = [0.9, 0.1, 0.05], 
                                milestone_steps = [0, len(train_x) * 100, len(train_x) * 350]
                            )
                        torch.set_grad_enabled(True)
                        #print(torch.set_grad_enabled)
                        
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
                        print('Training')
                        for epoch in range(1, 1500+1):
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
                            PC1_state = pc.state_dict()
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

   
                        print('after training')
                        print('flows')
                        
                        PC_state = pc.state_dict()
                        params = PC_state['params']
                        input_params =  PC_state['input_layer_group.layer_0.params']
                        #print('params = ',params)
                        #print(train_x)
                        #print('params = ',sum(params))
                        #print('Input_params = ', input_params)
                        print('params = ',params[params.nonzero()].squeeze(1))
                        print('input_params = ',input_params)#[input_params.nonzero()].squeeze(1))
                        Hess = HESSIAN(pc)
                        #print('Hessian')
                        #print(Hess)
                        #pc.zero_param_flows()
                        #print('params flows = ',pc.param_flows.detach().cpu())
                        x = train_x.to(device)
                        Hes = np.zeros([num_latents,num_latents])
                        for iii in range(len(x)):
                            pc.zero_param_flows()
                            ll = pc(torch.tensor(x[iii].reshape([1,dim])))
                            ll.mean().backward()
                            #pc.update_param_flows()
                            Hes = Hes + HESSIAN(pc)
                            #print(Hes)
                        print('Hessian')
                        #print((1/len(x))*Hes) 
                        eig_val = SGM((1/len(x))*Hes)   
                        min_eig = np.min(eig_val)
                        #eig_val = np.delete(eig_val,np.where(eig_val == np.min(eig_val)))
                        #EigVal = EigVal +  np.sort(eig_val)/trial
                        print('Eigenvalues = ', eig_val)
                        #dfs(ns)
                       


                        lls = pc(train_x.to(pc.device))
                        training_ll = lls.mean().detach().cpu().numpy().item()
                        print('Train LL = ',training_ll)
                        Train_ll_points.append(training_ll)
    
                        llst = pc(test_x.to(pc.device))
                        testing_ll = llst.mean().detach().cpu().numpy().item()
                        print('Test LL = ',testing_ll)
            
                        #print(train_x)
                        
                        #deriv = [()*Gauss(input_params.cpu()[0],input_params.cpu()[1],train_x[0][0])*Gauss(input_params.cpu()[2],input_params.cpu()[3],train_x[0][0]),(1/math.exp(training_ll))*Gauss(input_params.cpu()[4],input_params.cpu()[5],train_x[0][1])*Gauss(input_params.cpu()[6],input_params.cpu()[7],train_x[0][1])]                         
                        #print(deriv)
                        TEST_ll = llst.detach().cpu().numpy()
                        Test_ll_points.append(testing_ll)
                        #diff.append(abs(testing_ll-training_ll))
                        exp_test =[]
                        for x in TEST_ll:
                            exp_test.append(np.exp(x)) 
                         

                        Flatness.append(np.mean(eig_val))
                        
                   
                    avg_flat = avg_flat+np.array(Flatness)
                    
                    avg_diff = avg_diff+np.divide(np.array(Test_ll_points)-np.array(Train_ll_points),np.array(Test_ll_points))
                 
                """ 
                plt.figure(figsize=(25,20))
                plt.xlabel('Eigenvalues')
                plt.ylabel('Frequency')
                # Plotting
                plt.hist(EigVal)
                plt.title('Eigenspectrum of Hessian at local optima_' + 'latent_'+str(num_latents)+'_points_'+str(points))
                plt.savefig('Distribution of eigen values_'+str(num_latents)+'_'+ str(depth)+'_'+str(points))
                plt.close()     
                """
                # Figure size 
                plt.figure(figsize=(25,20))
                plt.xlabel('Peakiness')
                plt.ylabel('likelyhood')
                # Plotting
                sort = np.argsort(avg_flat)
                plt.plot((avg_flat/trial)[sort],(avg_diff/trial)[sort], marker = 'o', label=' difference Likelihood')
                plt.title('Difference in likelihood vs flat' + '_'+str(num_latents))
                plt.savefig('avg Train,Test vs flat_geom_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
                plt.close()     
                

                # Figure size
                plt.figure(figsize=(25,20))
                plt.xlabel('Points')
                plt.ylabel('likelyhood')
                # Plotting
                plt.plot(Points,avg_diff/trial, marker = 'o', label=' difference Likelihood')
                plt.title('Difference in likelihood vs point' + '_'+str(num_latents))
                plt.savefig('Difference in likelihood vs Points_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
                plt.close()     
               
                Flat.append(avg_flat/trial)
                diff_.append(avg_diff/trial)
               
           
            # Figure size 
            plt.figure(figsize=(30,25)) 
            plt.xlabel('Peakiness')
            plt.ylabel('Degree of overfitting')
            # Plotting
            sort = np.argsort(Flat[0])
            plt.plot(Flat[0][sort],diff_[0][sort], marker = 'o',markersize = 20, label= 'Latent size = '+str(Latents[0]))
            sort = np.argsort(Flat[1])
            plt.plot(Flat[1][sort],diff_[1][sort], marker = 'o',markersize = 20, label= 'Latent size = '+str(Latents[1]))
            sort = np.argsort(Flat[2])
            plt.plot(Flat[2][sort],diff_[2][sort], marker = 'o',markersize = 20, label= 'Latent size = '+str(Latents[2]))
            plt.title('Degree of overfitting vs Peakiness(across different latent size)_'+'_dim = '+str(dim))
            
            plt.legend()
            plt.savefig('avg Degree of overfitting vs Peakiness_geom1_'+str(dim)+'_'+ str(depth)+'_'+ str(num_repetitions))
            plt.close()     
               
         
            
            # Figure size
            plt.figure(figsize=(30,25))
            plt.xlabel('Points')
            plt.ylabel('Degree of overfitting')
            # Plotting
            plt.plot(Points,diff_[0], marker = 'o',markersize = 20, label= 'Latent size = '+str(Latents[0]))
            plt.plot(Points,diff_[1], marker = 'o',markersize = 20, label= 'Latent size = '+str(Latents[1]))
            plt.plot(Points,diff_[2], marker = 'o',markersize = 20, label= 'Latent size = '+str(Latents[2]))
            plt.title('Degree of overfitting vs point (across different latent size)'+'_dim = '+ str(dim))
            plt.xticks(Points)
            plt.legend()
            plt.savefig('Degree of overfitting vs Points_'+str(dim)+'_'+ str(depth)+'_'+ str(num_repetitions))
            plt.close()      
            
            # Figure size
            plt.figure(figsize=(30,25))
            plt.xlabel('Points')
            plt.ylabel('Peakiness')
            # Plotting
            plt.plot(Points,Flat[0], marker = 'o',markersize = 20, label= 'Latent size = '+str(Latents[0]))
            plt.plot(Points,Flat[1], marker = 'o',markersize = 20, label= 'Latent size = '+str(Latents[1]))
            plt.plot(Points,Flat[2], marker = 'o',markersize = 20, label= 'Latent size = '+str(Latents[2]))
            plt.title('Peakiness vs point (across different latent size)'+'_dim = '+ str(dim))
            plt.xticks(Points)
            plt.legend()
            plt.savefig('Peakiness vs Points_'+str(dim)+'_'+ str(depth)+'_'+ str(num_repetitions))
            plt.close()              
