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
from scipy.stats import norm
import matplotlib
import pandas as pd

matplotlib.rcParams.update({'font.size': 22})
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)
device = torch.device(device)
batchsize = 1

mu1,mu2,mu3 = 5,3.5,7
sigma1,sigma2,sigma3 = 0.2,0.5,0.3

Latents = [2,3,4]
depth = 0
num_repetitions = 1
instance = 10

#print(mu,sigma)


x_orig = []
for i in range(2000):
    x_orig.append(np.random.normal(mu1,sigma1,[1,2]))
for i in range(1000):
    x_orig.append(np.random.normal(mu2,sigma2,[1,2]))
for i in range(1000):
    x_orig.append(np.random.normal(mu3,sigma3,[1,2]))   
    
        

x_test = []
for i in range(10):
    x_test.append(np.random.normal(mu1,sigma1,[1,2]))
for i in range(10):
    x_test.append(np.random.normal(mu2,sigma2,[1,2]))
for i in range(10):
    x_test.append(np.random.normal(mu3,sigma3,[1,2])) 

x_orig = np.array(x_orig)   
x_orig = np.sort(x_orig)  
X_orig = x_orig.reshape([len(x_orig),2]) 

test_x_orig = np.array(x_test)   
test_x_orig = np.sort(test_x_orig)
test_x = test_x_orig.reshape([len(test_x_orig),2])


Ground_truth1 = norm.pdf(X_orig,mu1,sigma1)
Ground_truth2 = norm.pdf(X_orig,mu2,sigma2)
Ground_truth3 = norm.pdf(X_orig,mu3,sigma3)
Points = np.array([500])

qwert = 1
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
            if qwert ==1:
                plt.figure()
                juice_vis.plot_pc(ns, node_id = True, node_num_label = True)
                plt.savefig('SPN_Gaussian_'+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
                qwert =2         
            optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1, method = "EM")
            scheduler = juice.optim.CircuitScheduler(
                    optimizer, 
                    method = "multi_linear", 
                    lrs = [0.9, 0.9, 0.9], 
                    milestone_steps = [0, len(train_x) * 100, len(train_x) * 1500]
                )

            #print('before training')
            PC_state = pc.state_dict()
            params = PC_state['params']
            #for key in PC_state.keys():
            #    print(key)
            ii = np.sum(np.array(params.cpu())==0)
            input_params =  PC_state['input_layer_group.layer_0.params']
            print('params = ',params[ii:])
            print('sum params = ',sum(params[ii:]))            
            #print('Input_params = ', input_params)
            lls = pc(train_x.to(pc.device))
            print('Train LL = ',lls.mean().detach().cpu().numpy().item())
    
            """
            Train_ll = []
            Test_ll =[]
            start_time = time.time()
            for epoch in range(1, 50+1):
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
 
            plt.figure(figsize = (15,12))
            plt.plot(Train_ll, label='Likelihood')
            plt.xlabel('a')
            plt.ylabel('Likelihood')
            plt.title('Likelihood vs Epochs')
            plt.legend()
            plt.savefig('Likelihood_epoch_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
            plt.close()    
          
            exp_train =[]
            for x in TRAIN_ll:
                exp_train.append(np.exp(x)) 

   
            print('after training')
            PC_state = pc.state_dict()
            params = PC_state['params']
            input_params =  PC_state['input_layer_group.layer_0.params']
            #print('params = ',params)
            #print('sum params = ',sum(params))
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
 
            pc_toy = juice.compile(ns)
            pc_toy.to(device)
            PC_state = pc.state_dict()
            #print(PC_state)
            params = PC_state['params'].cpu()
          
            direction = np.zeros(len(params))
              
            ii = np.sum(np.array(params)==0)
            
            print('variable  = ',len(params)-ii)
            for i in range(ii,int(len(params))-1,2):
                m = min(params[i],params[i+1])
                d = np.random.normal(m/2,m/6)
                direction[i] = d
                direction[i+1] = -d    
            #lls = pc_toy(train_x.to(pc.device))
            #print('Test LL new = ',lls.mean().detach().cpu().numpy().item())
            PC_train_ll = []
            A = np.linspace(-0.5,0.5,100)
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
            num_sum = (depth**2+depth)*num_latents+1
            start = ii
            for sum_node in range(1,num_sum+1):
                end = num_latents
                 
                for i in range(start,start+end-1,2):
                 
                    m = min(params[i],params[i+1])
                    d = np.random.normal(m/2,m/10)
                    norm_direction[i] = d
                    norm_direction[i+1] = -d 
                norm1 = np.linalg.norm(norm_direction[start:start+end])
                norm2 = np.linalg.norm(params[start:start+end]) 
                norm_direction = (np.array(norm_direction)/norm1)*norm2
                start = start+end
            PC_train_ll_norm = []    
            for a in A:
                PC_state['params'] = params.to(pc.device)+a*direction
                pc_toy.load_state_dict(PC_state)
                  
                x = train_x.to(pc.device)
                lls = pc_toy(x)
                train_ll = lls.mean().detach().cpu().numpy().item()
                                   
                PC_train_ll_norm.append(train_ll) 
            
            f1 = abs(PC_train_ll[0]-max(PC_train_ll))
            f2 = abs(PC_train_ll[-1]-max(PC_train_ll))
            norm_flatness.append(max(f1,f2))    
            """
        """            
        plt.figure(figsize = (15,12))
        plt.plot(A,PC_train_ll, label='Likelihood')
        plt.xlabel('a')
        plt.ylabel('Likelihood')
        plt.title('Likelihood vs A')
        plt.legend()
        plt.savefig('Likelihood_geometry_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
        plt.close()  
        
        plt.figure(figsize = (15,12))
        plt.plot(A,PC_train_ll_norm, label='Likelihood')
        plt.xlabel('a')
        plt.ylabel('Likelihood')
        plt.title('Likelihood vs A')
        plt.legend()
        plt.savefig('Likelihood_geometry_norm_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
        plt.close()   
        """ 
