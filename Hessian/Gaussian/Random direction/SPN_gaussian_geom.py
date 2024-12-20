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
import seaborn as sns

matplotlib.rcParams.update({'font.size': 35})
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)
device = torch.device(device)
batchsize = 1

mu1,mu2,mu3 = 5,3.5,7
sigma1,sigma2,sigma3 = 0.2,0.5,0.3

Latents = [50]
depth = 0
num_repetitions = 1
Points = [300,500,1000,2000,5000,10000]
#Points_ = [10+5,2000+5]
points_str = []
for x in Points:
    points_str.append(str(x))
#print(mu,sigma)


x_orig = []
for i in range(50000):
    x_orig.append(np.random.normal(mu1,sigma1))
for i in range(25000):
    x_orig.append(np.random.normal(mu2,sigma2))
for i in range(25000):
    x_orig.append(np.random.normal(mu3,sigma3))         

x_orig = np.array(x_orig)   
x_orig1 = np.sort(x_orig)  
X_orig = x_orig1.reshape([len(x_orig1),1]) 

test_x_orig = np.random.choice(x_orig1, 500)
test_x = test_x_orig.reshape([len(test_x_orig),1])

Ground_truth1 = norm.pdf(X_orig,mu1,sigma1)
Ground_truth2 = norm.pdf(X_orig,mu2,sigma2)
Ground_truth3 = norm.pdf(X_orig,mu3,sigma3)

Ground_truth_test1 = norm.pdf(test_x,mu1,sigma1)
Ground_truth_test2 = norm.pdf(test_x,mu2,sigma2)
Ground_truth_test3 = norm.pdf(test_x,mu3,sigma3)

TEST_Truth = Ground_truth_test1/2+Ground_truth_test2/4+Ground_truth_test3/4

rer = 1
Deg_Over = []
NORM_Flatness = []
Peakness = []
for num_latents in Latents:

    train_average = []
    test_average = []

    Train_ll_points =[]
    Test_ll_points =[]
    #X_orig = np.random.normal(mu1,sigma1,[1000,1])
    #Points = np.array([50,200,500,800,1000,2000])
    Flatness =[]   
    norm_flatness = []
    deg_overfitting = []
    for points in Points:

        train_x_orig = np.random.choice(x_orig,points)
        train_x = train_x_orig.reshape([len(train_x_orig),1])
    
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
        input_params1 =  PC_state['input_layer_group.layer_0.params'].cpu().tolist() 
        #print('params = ',params)
        #print('Input_params = ', input_params)
        lls = pc(train_x.to(pc.device))
        #print('Train LL = ',lls.mean().detach().cpu().numpy().item())
 
      
        sigma_gen = []
       
        for i in range(len(input_params1)):
            if i%2 != 0:
                sigma_gen.append(input_params1[i])
 
        plt.figure(figsize = (35,25))    
        plt.hist(sigma_gen,bins=4,rwidth = 0.5) 
        #plt.plot(sorted(train_x_orig),sorted_density,label = 'Learned distribution',color = 'red')
        plt.xlabel('Variance')
        plt.ylabel('Count')
        plt.title('Generated Distribution variance('+str(points)+' pts)_before_0.3' + '_'+str(num_latents))   
        #plt.legend()
        plt.savefig('Generated Distribution variance('+str(points)+' pts)_before_03' + '_'+str(num_latents))
        plt.close()    

        Train_ll = []
        Test_ll =[]
        start_time = time.time()
        for epoch in range(1, 2000+1):
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
        input_params1 =  PC_state['input_layer_group.layer_0.params'].cpu().tolist() 
        print('params = ',params)
        #print('params = ',sum(params))
        print('Input_params = ', input_params)
        lls = pc(train_x.to(pc.device))
        training_ll = lls.mean().detach().cpu().numpy().item()
        print('Train LL = ',training_ll)
        Train_ll_points.append(training_ll)
    
        llst = pc(test_x.to(pc.device))
        testing_ll = llst.mean().detach().cpu().numpy().item()
        print('Test LL = ',testing_ll)
        TEST_ll = llst.detach().cpu().numpy()
        Test_ll_points.append(testing_ll)
        exp_test =[]
        for x in TEST_ll:
            exp_test.append(np.exp(x))     
   
        sort = np.argsort(train_x_orig)
        sorted_density = np.array(exp_train)[sort]
        plt.figure(figsize = (35,25))   
        plt.plot(X_orig,Ground_truth1/2+Ground_truth2/4+Ground_truth3/4,label = 'Ground Truth') 
        #plt.plot(sorted(train_x_orig),sorted_density,label = 'Learned distribution',color = 'red')
        plt.scatter(train_x,exp_train,label = 'Learned distribution',color = 'red',s = 0.5)  
        plt.ylabel('Likelihood')
        plt.xlabel('X')
        plt.title('Train_density('+str(points)+'pts)' + '_'+str(num_latents))   
        plt.legend()
        plt.savefig('Train_density_Gaussian_'+str(points)+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
        plt.close()     
        
        params1 = PC_state['params'].cpu().tolist()    
        ii = np.sum(np.array(params1)==0)
        
        mu_gen = []
        sigma_gen = []
        P_gen = []
        for i in range(len(input_params1)):
            if i%2 == 0:
                mu_gen.append(input_params1[i])
            else:
                sigma_gen.append(input_params1[i])
 
        plt.figure(figsize = (35,25))    
        plt.hist(sigma_gen,bins=4,rwidth = 0.5) 
        #plt.plot(sorted(train_x_orig),sorted_density,label = 'Learned distribution',color = 'red')
        plt.xlabel('Variance')
        plt.ylabel('Count')
        plt.title('Generated Distribution variance('+str(points)+' pts)_after_0.3' + '_'+str(num_latents))   
        #plt.legend()
        plt.savefig('Generated Distribution variance('+str(points)+' pts)_after_03' + '_'+str(num_latents))
        plt.close()  
                
        for i in range(len(params1[ii:])): 
            P_gen.append(params1[ii+i])
        
        x_gen = []
        for i in range(num_latents):
            for j in range(int(1000*P_gen[i])):
                x_gen.append(np.random.normal(mu_gen[i],sigma_gen[i]))
            

        x_gen = np.array(x_gen)   
        x_gen = np.sort(x_gen)  
        X_gen = x_gen.reshape([len(x_gen),1]) 
        
        Gen_Data = 0 
        for i in range(num_latents):
            Generate_ = norm.pdf(X_gen,mu_gen[i],sigma_gen[i])
            Gen_Data = Gen_Data + Generate_*P_gen[i]
              
        plt.figure(figsize = (35,25))    
        plt.plot(X_gen,Gen_Data,label = 'Generated Distribution') 
        #plt.plot(sorted(train_x_orig),sorted_density,label = 'Learned distribution',color = 'red')
        plt.ylabel('Likelihood')
        plt.xlabel('X')
        plt.title('Train_density('+str(points)+'pts)' + '_'+str(num_latents))   
        plt.legend()
        plt.savefig('Generated_Train_density_Gaussian_'+str(points)+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
        plt.close()  

        pc_toy = juice.compile(ns)
        pc_toy.to(device)
        PC_state = pc.state_dict()
        #print(PC_state)
        params = PC_state['params'].cpu()         
            

        #print(ii)
        direction = np.zeros(len(params))
        for i in range(ii,int(len(params))-1,2):
            m = min(params[i],params[i+1])
            d1 = np.random.normal(m/6,m/6)
            #d2 = np.random.normal(m/2,m/2.5)
            direction[i] = d1
            direction[i+1] = -d1
        direction = torch.tensor(np.array(direction)).to(pc.device)    
        rer = 0    
        #norm1 = np.linalg.norm(direction[ii:])
        #direction = (np.array(direction)/norm1)            
              
        #lls = pc_toy(train_x.to(pc.device))
        #print('Test LL new = ',lls.mean().detach().cpu().numpy().item())
        PC_train_ll = []
        A = np.linspace(-0.5,0.5,100)
        
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
         
        """
        #print(PC_train_loss)  
        plt.figure(figsize = (35,25))
        plt.plot(A,PC_train_ll, label='Likelihood')
        plt.xlabel('a')
        plt.ylabel('Likelihood')
        plt.title('Likelihood vs A_'+str(num_latents)+'_'+str(points)+'_Peakness = '+str(np.round(max(f1,f2),6))+'\n'+'Degree of overfitting = '+str(abs(np.round((testing_ll - training_ll)/testing_ll,4)))) #+ '\n' + 'Mixing Weights = '+ str(params[ii:])+'\n'+'Input params = '+str(torch.round(input_params[0:],decimals =4)))
        plt.legend()
        plt.savefig('Likelihood_geometry_Gaussian_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions) +'_'+str(points))
        plt.close()   
        """
        norm_direction = np.zeros(len(params))
            
        for i in range(ii,int(len(params))-1,2):
            m = min(params[i],params[i+1])
            d = np.random.normal(m/6,m/6)
            norm_direction[i] = d
            norm_direction[i+1] = -d 
        norm1 = np.linalg.norm(norm_direction[ii:])
        norm2 = np.linalg.norm(params[ii:]) 
        norm_direction = (np.array(norm_direction)/norm1)*norm2

          
        PC_train_ll_norm = []
        norm_direction = torch.tensor(norm_direction).to(pc.device)
        #print(norm_direction)
        #print('direction = ',direction)
        A_ = np.linspace(-0.1,0.1,100)
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
        norm_flatness.append(f1+f2)
        
        deg_overfitting.append(abs((testing_ll - training_ll)/testing_ll))
        """
        plt.figure(figsize = (35,25))
        plt.plot(A,PC_train_ll_norm, label='Likelihood')
        plt.xlabel('a')
        plt.ylabel('Likelihood')
        plt.title('Likelihood vs A_'+str(num_latents)+'_'+str(points)+'_Peakness = '+str(np.round(f1+f2,6))+'\n'+'Degree of overfitting = '+str(abs(np.round((testing_ll - training_ll)/testing_ll,4)))) #+ '\n' + 'Mixing Weights = '+ str(params[ii:])+'\n'+'Input params = '+str(torch.round(input_params[0:],decimals =4)))
        plt.legend()
        plt.savefig('Likelihood_geometry_Gaussian_norm_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions) +'_'+str(points))
        plt.close()
        """  
    Deg_Over.append(deg_overfitting)
    NORM_Flatness.append(norm_flatness)
    Peakness.append(Flatness)
    """    
    # Figure size
    plt.figure(figsize=(35,25))
    
    plt.xlabel('Points')
    plt.ylabel('likelihood')
    # Plotting 
    plt.scatter(Points, Train_ll_points , label=' Train Likelihood')
    plt.scatter(Points, Test_ll_points, label='Test_likelihood')
    plt.legend(loc='upper right')
    plt.title('Train vs Test likelihood' + '_'+str(num_latents))
    plt.savefig('Train vs Test likelihood_geom_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
    plt.close()    
 
    # Numbers of pairs of bars you want
    N = 2

    # Data on X-axis

    # Specify the values of blue bars (height)
    blue_bar = abs(np.array(Train_ll_points)-np.array(Test_ll_points))
    # Specify the values of orange bars (height)
    #orange_bar = np.array(Test_likely)*-1

    # Position of bars on x-axis
    ind = np.arange(N)
    """

    # Figure size
    plt.figure(figsize = (35,25))
    fig, ax1 = plt.subplots()
    fig.set_figheight(25)
    # set width of each subplot as 8
    fig.set_figwidth(35)
    color = 'blue' 
    # Width of a bar 
    #width = 0.2       
    ax1.set_xlabel('Training size')
    ax1.set_ylabel('Peakness', color=color)
    # Plotting
    ax1.plot(Points,norm_flatness,label = 'Normalized direction',color=color,marker = 'o', markersize=20)
    #ax1.plot(Points,deg_overfitting,label ='Degree of Overfitting',color = 'red',marker = 'o', markersize=10)
    
    # ax1.bar(ind + width, orange_bar, width, label='Test Likelihood')
    ax1.legend(loc='upper left')
    color = 'red' 
    ax2 = ax1.twinx()
    ax2.set_ylabel('Degree of Overfitting', color=color)
    ax2.plot(Points,deg_overfitting,label ='Degree of Overfitting',color = 'red',marker = 'o', markersize=10)
    ax2.legend(loc='upper right')
    plt.title('Degree of Overfitting and norm flatness')
    plt.savefig('Degree of Overfitting and norm flatness_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
    
    # Figure size
    plt.figure(figsize = (35,25))
    fig, ax1 = plt.subplots()
    fig.set_figheight(25)
    # set width of each subplot as 8
    fig.set_figwidth(35)
    color = 'blue' 
    # Width of a bar 
    #width = 0.2       
    ax1.set_xlabel('Training size')
    ax1.set_ylabel('Peakness', color=color)
    # Plotting
    ax1.plot(Points,Flatness,label = 'Random direction',color=color,marker = 'o', markersize=20)
    #ax1.plot(Points,deg_overfitting,label ='Degree of Overfitting',color = 'red',marker = 'o', markersize=10)
    
    # ax1.bar(ind + width, orange_bar, width, label='Test Likelihood')
    ax1.legend(loc='upper left')
    color = 'red' 
    ax2 = ax1.twinx()
    ax2.set_ylabel('Degree of Overfitting', color=color)
    ax2.plot(Points,deg_overfitting,label ='Degree of Overfitting',color = 'red',marker = 'o', markersize=10)
    ax2.legend(loc='upper right')
    plt.title('Degree of Overfitting and flatness')
    plt.savefig('Degree of Overfitting and flatness_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))    
    
    """
    # Figure size
    plt.figure(figsize = (35,25))
    plt.xlabel('Peakness')
    plt.ylabel('Degree of overfitting')
    # Plotting
    plt.plot(Flatness, deg_overfitting, label='Latents = 10',marker = 'o', markersize=15)
    plt.legend()
    plt.title('Degree of Overfitting vs flatness' + '_'+str(num_latents))
    plt.savefig('Degree of Overfitting vs flatness_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
    plt.close()     

    # Figure size
    plt.figure(figsize = (35,25))
    plt.xlabel('norm_Peakness')
    plt.ylabel('Degree of overfitting')
    # Plotting
    plt.plot(norm_flatness, deg_overfitting, label='Latents = 10',marker = 'o', markersize=15)
    plt.legend()
    plt.title('Degree of Overfitting vs norm flatness' + '_'+str(num_latents))
    plt.savefig('Degree of Overfitting vs norm flatness_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
    plt.close() 
    """

# Figure size
plt.figure(figsize = (35,25))
plt.xlabel('Training size')
plt.ylabel('Degree of overfitting')
# Plotting
plt.plot(Points,Deg_Over[0] , label='Latents = 50',marker = 'o', markersize=15)
#plt.plot(Points,Deg_Over[1] , label='Latents = 50',marker = 'o', markersize=15)
plt.title('Degree of Overfitting vs Train size _init=0.3')
plt.legend()
plt.savefig('Degree of Overfitting vs Train size_across latents_03')
plt.close()     
"""
# Figure size
plt.figure(figsize = (35,25))
plt.xlabel('norm_Peakness')
plt.ylabel('Degree of overfitting')
# Plotting
plt.plot(NORM_Flatness[0], Deg_Over[0] , label='Latents = 10',marker = 'o', markersize=15)
plt.plot(NORM_Flatness[1], Deg_Over[1] , label='Latents = 20',marker = 'o', markersize=15)
plt.plot(NORM_Flatness[2], Deg_Over[2] , label='Latents = 30',marker = 'o', markersize=15)
plt.plot(NORM_Flatness[3], Deg_Over[3] , label='Latents = 50',marker = 'o', markersize=15)
plt.legend()
plt.title('Degree of Overfitting vs norm flatness')
plt.savefig('Degree of Overfitting vs norm flatness_across latents')
plt.close()     
   
# Figure size
plt.figure(figsize = (35,25))
fig, ax1 = plt.subplots()
fig.set_figheight(25)
# set width of each subplot as 8
fig.set_figwidth(35)
color = 'blue' 
# Width of a bar 
#width = 0.2       
ax1.set_xlabel('Training size')
ax1.set_ylabel('Peakness', color=color)
# Plotting
ax1.plot(Points,Flatness,label = 'Un normalized direction',color='black')
ax1.plot(Points,norm_flatness,label = 'Normalized direction',color='blue')
# ax1.bar(ind + width, orange_bar, width, label='Test Likelihood')
ax1.legend(loc='upper left')
color = 'red' 
ax2 = ax1.twinx()
ax2.set_ylabel('Degree of Overfitting', color=color)
ax2.plot(Points,deg_overfitting,label ='Degree of Overfitting',color = 'red' )
ax2.legend(loc='upper right')
plt.title('Degree of Overfitting and flatness')
plt.savefig('Degree of Overfitting and flatness_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
plt.close()    
"""
