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

DIM = [2]

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

    ground_pdf_1 = distr1.pdf(test_x)
    ground_pdf_2 = distr2.pdf(test_x)
    ground_pdf_3 = distr3.pdf(test_x)

    TEST_Truth = ground_pdf_1/2+ground_pdf_2/4+ground_pdf_3/4

    Points = np.array([100,250,500,800,1000,1500,2000,3000])
    points_str = []
    for x in Points:
        points_str.append(str(x))
    for depth in Depth:
        qwert = 1
        for Latents in LATENTS:

            Flat =[]
            norm_flat = []
            diff_ = []
            TV_dist = []
            for num_latents in Latents:

                train_average = []
                test_average = []
 
 
                #X_orig = np.random.normal(mu1,sigma1,[1000,1])
                avg_diff = np.zeros(len(Points))
                avg_flat = np.zeros(len(Points))
                avg_normflat = np.zeros(len(Points))
                avg_TV_dist = np.zeros(len(Points))
                for trial in range(100):
                    Train_ll_points =[]
                    Test_ll_points =[]
                    TV_dist_points = []
                    Flatness =[]   
                    norm_flatness = []
                    for points in Points:

                        train_x_random = x_orig
                        np.random.shuffle(train_x_random)
                        train_x = train_x_random[:points]
                        train_x = train_x.reshape([len(train_x),dim])
 
    
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
                            plt.savefig('SPN_Gaussian_'+'_dim = '+str(dim)+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
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

   
                        print('after training')
                        PC_state = pc.state_dict()
                        params = PC_state['params']
                        input_params =  PC_state['input_layer_group.layer_0.params']
                        #print('params = ',params)
                        #print('params = ',sum(params))
                        #print('Input_params = ', input_params)
                        lls = pc(train_x.to(pc.device))
                        training_ll = lls.mean().detach().cpu().numpy().item()
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
                        tv= 0
                        for i in range(len(exp_test)):
                            tv = tv + abs(exp_test[i]- TEST_Truth[i])
                        TV_dist_points.append(tv[0]) 
                    avg_diff = avg_diff+np.array(abs(np.array(Train_ll_points)-np.array(Test_ll_points)))
                    avg_TV_dist = avg_TV_dist+np.array(TV_dist_points)
    

                # Figure size
                plt.figure(figsize=(15,12))
                plt.xlabel('Points')
                plt.ylabel('likelihood')
                # Plotting
                plt.scatter(Points,avg_diff/100 , label=' difference Likelihood')
                plt.title('Difference in likelihood vs point' +'_dim = '+str(dim)+ '_'+str(num_latents)+'_'+ str(depth))
                plt.legend(loc = "upper right")
                plt.savefig('Difference in likelihood vs Points'+'_dim = '+str(dim)+'_'+str(num_latents)+'_'+ str(depth))
                plt.close()     

                diff_.append(avg_diff/100)
  
                # Numbers of pairs of bars you want
                N = len(Points)

                # Data on X-axis

                # Specify the values of blue bars (height)
                blue_bar = avg_diff/100
                # Specify the values of orange bars (height)
                #orange_bar = np.array(Test_likely)*-1

                # Position of bars on x-axis
                ind = np.arange(N)

                # Figure size
                plt.figure(figsize=(15,12))
                fig, ax1 = plt.subplots()
                fig.set_figheight(12)
 
                # set width of each subplot as 8
                fig.set_figwidth(15)
                color = 'blue' 
                # Width of a bar 
                width = 0.2       
                ax1.set_xlabel('Training size')
                ax1.set_ylabel(' likelyhood', color=color)
                # Plotting
                ax1.bar(ind, blue_bar , width, label='difference in Likelihood')
                ax1.set_xticks(list(np.arange(N)))
                ax1.set_xticklabels(points_str)
                # ax1.bar(ind + width, orange_bar, width, label='Test Likelihood')
                ax1.legend(loc='upper left')
                color = 'red'
                ax2 = ax1.twinx()
                ax2.set_ylabel('Total variance test', color=color)
                ax2.bar(ind+width,  avg_TV_dist/100 , width, label='Total variance in test',color=color)
                ax2.legend(loc='upper right')
                plt.title('Likelihood difference and Total variance'+'_dim = '+str(dim)+'_'+str(num_latents))
                plt.savefig('Likelihood and Total variance_'+'_dim = '+str(dim)+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
                plt.close()    
 
            # Figure size
            plt.figure(figsize=(15,12))
            plt.xlabel('Points')
            plt.ylabel('likelihood')
            # Plotting
            plt.scatter(Points,diff_[0], label= 'Latent size = '+str(Latents[0]))
            plt.scatter(Points,diff_[1], label= 'Latent size = '+str(Latents[1]))
            plt.scatter(Points,diff_[2], label= 'Latent size = '+str(Latents[2]))
            plt.title('Difference in likelihood vs point (across different latent size)'+'_dim = '+str(dim)+'_'+ str(depth))
            plt.legend(loc = "upper right")
            plt.savefig('Difference in likelihood vs Points_' +str(dim)+'_'+str(Latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
            plt.close()      
