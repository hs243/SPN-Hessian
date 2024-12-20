import pyjuice as juice
import pyjuice.visualize as juice_vis
import torch
import torchvision
import time
import pyjuice.nodes.distributions as dists
import datasets
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

print(device)
device = torch.device(device)
batchsize = 1200

dataset = 'nltcs'
num_latents = 20
Depths = [1,2,3,5,7,9]
num_repetitions = 2
instance = 10

print(dataset)

train_average = []
test_average = []
train_x_orig, test_x_orig, valid_x_orig = datasets.load_debd(dataset, dtype='float32')

train_x = train_x_orig
test_x = test_x_orig
valid_x = valid_x_orig
#print(train_x)
# to torch
train_x = torch.tensor(train_x,dtype=torch.uint8)
valid_x = torch.tensor(valid_x,dtype=torch.uint8)
test_x = torch.tensor(test_x,dtype=torch.uint8)
#print(train_x)
train_N, num_dims = train_x.shape
valid_N = valid_x.shape[0]
test_N = test_x.shape[0]

train_loader = DataLoader(
        dataset = TensorDataset(train_x),
        batch_size = batchsize,
        shuffle = True,
        drop_last = True
    )
    
valid_loader = DataLoader(
        dataset = TensorDataset(valid_x),
        batch_size = batchsize,
        shuffle = False,
        drop_last = True
    )

test_loader = DataLoader(
        dataset = TensorDataset(test_x),
        batch_size = batchsize,
        shuffle = False,
        drop_last = True
    )


for depth in Depths:
    print(depth)
    flatness = []
    norm_flatness = []
    ns = juice.structures.RAT_SPN(num_vars = len(train_x[0]), 
            num_latents = num_latents,
            depth = depth,
            num_repetitions = num_repetitions,
            input_node_params= {"num_cats": 2}
        )
    print(ns)
    pc = juice.compile(ns)

    pc.to(device)
           
    
    plt.figure()
    juice_vis.plot_pc(ns, node_id = True, node_num_label = True)
    plt.savefig('SPN'+dataset+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))

    Train_likely =[]
    Test_likely =[]
        
    for ins in range(instance):
        pc = juice.compile(ns)

        pc.to(device)
        optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1, method = "EM")
        scheduler = juice.optim.CircuitScheduler(
                optimizer, 
                method = "multi_linear", 
                lrs = [0.9, 0.1, 0.05], 
                milestone_steps = [0, len(train_x) * 100, len(train_x) * 350]
            )


        PC_state = pc.state_dict()
    
        params =  PC_state['input_layer_group.layer_0.params']
        #print(params)
        lls = pc(train_x.to(pc.device))
        print('Train LL = ',lls.mean().detach().cpu().numpy().item())

        #print(len(params))
        #print(sum(params))

        #print(direction)
        #print(len(direction))    

        #plt.figure()
        #juice_vis.plot_pc(ns, node_id = True, node_num_label = True)
        #plt.savefig('SPN'+str(num_latents)+'_'+dataset)

        Train_ll = []
        Test_ll =[]
        start_time = time.time()
        for epoch in range(1, 10+1):
            t0 = time.time()
            train_ll = 0.0
    
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
            Train_ll.append(train_ll)
            t1 = time.time()       
    
            test_ll = 0.0
            for batch in test_loader:
                x = batch[0].to(pc.device)
                lls = pc(x)
                test_ll += lls.mean().detach().cpu().numpy().item()
    
            test_ll /= len(test_loader)
            Test_ll.append(test_ll)
                 
            valid_ll = 0.0
            for batch in valid_loader:
                x = batch[0].to(pc.device)
                lls = pc(x)
                valid_ll += lls.mean().detach().cpu().numpy().item()
            valid_ll /= len(valid_loader)
            t2 = time.time()    
           
        Train_likely.append(Train_ll[-1])  
        Test_likely.append(Test_ll[-1])
       
        lls = pc(train_x.to(pc.device))
        train_LL = lls.mean().detach().cpu().numpy().item()
        print('Train LL = ',train_LL)

        pc_toy = juice.compile(ns)
        pc_toy.to(device)
        PC_state = pc.state_dict()
        #print(PC_state)
        params =  PC_state['input_layer_group.layer_0.params'].cpu()
        #print('parameters = ',params)
        direction = []
        for i in range(0,int(len(params)),2):
            m = min(params[i],params[i+1])
            d = np.random.normal(m/2,m/2)
            direction.append(d)
            direction.append(-d)      
        #lls = pc_toy(train_x.to(pc.device))
        #print('Test LL new = ',lls.mean().detach().cpu().numpy().item())
        PC_train_ll = []
        A = np.linspace(-0.3,0.3,100)
        direction = torch.tensor(np.array(direction)).to(pc.device)
        #print('direction = ',direction)
        for a in A:
            PC_state['input_layer_group.layer_0.params'] = params.to(pc.device)+a*direction
            pc_toy.load_state_dict(PC_state)
                      
            x = train_x.to(pc.device)
            lls = pc_toy(x)
            train_ll = lls.mean().detach().cpu().numpy().item()
                                   
            PC_train_ll.append(train_ll) 
            
        
        #print(PC_train_loss)  
        plt.figure(figsize = (15,12))
        plt.plot(A,PC_train_ll, label='Likelihood')
        plt.xlabel('a')
        plt.ylabel('Likelihood')
        plt.title('Likelihood vs A')
        plt.legend()
        plt.savefig('Likelihood_geometry_4' + dataset+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions) +'_'+str(ins))
        plt.close()
        
        f1 = train_LL - PC_train_ll[0]
        f2 = train_LL - PC_train_ll[-1]
        flatness.append(f1+f2)
        """
        norm_direction = []
        for i in range(0,int(len(params)),2):
            m = min(params[i],params[i+1])
            d = np.random.normal(m/2,m/2)
            D = np.array([d,-d])
            norm1 = np.linalg.norm(D)
            norm2 = np.linalg.norm([params[i],params[i+1]]) 
            D = (D/norm1)*norm2
            norm_direction.append(D[0]*0.005)
            norm_direction.append(D[1]*0.005)
            
        PC_train_ll_norm = []
        norm_direction = torch.tensor(np.array(norm_direction)).to(pc.device)
        print(norm_direction)
        print('direction = ',direction)
        A_ = np.linspace(-0.05,0.05,100)
        for a in A_:
            PC_state['input_layer_group.layer_0.params'] = params.to(pc.device)+a*norm_direction
            pc_toy.load_state_dict(PC_state)
                      
            x = train_x.to(pc.device)
            lls = pc_toy(x)
            train_ll = lls.mean().detach().cpu().numpy().item()
                                
            PC_train_ll_norm.append(train_ll)
 
        plt.figure(figsize = (15,12))
        plt.plot(A_,PC_train_ll_norm, label='Likelihood')
        plt.xlabel('a')
        plt.ylabel('Likelihood')
        plt.title('Likelihood vs A')
        plt.legend()
        plt.savefig('Likelihood_geometry_norm_4' + dataset+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions) +'_'+str(ins))
        plt.close()    
     
        f1 = train_LL - PC_train_ll_norm[0]
        f2 = train_LL - PC_train_ll_norm[-1]
        norm_flatness.append(f1+f2)        
        """        
        #print(f"[Epoch {epoch}/{10}][train LL: {train_ll:.2f}; val LL: {valid_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; val forward {t2-t1:.2f}] ")

    train_avg = [sum(Train_likely)/len(Train_likely)]
    test_avg = [sum(Test_likely)/len(Test_likely)]
    train_average.append(train_avg)
    test_average.append(test_avg)
  
    train_avg = train_avg*instance
    test_avg = test_avg*instance

    plt.figure(figsize = (15,12))
    #fig, ax1 = plt.subplots()
    color = 'tab:red'
    #ax1.set_xlabel('training instance')
    #ax1.set_ylabel('Likelihood')
    plt.plot(Train_likely, label='Train Likelihood',color = 'red')
    #plt.plot(val_likely, label='Valid Likelihood',color = 'blue')
    plt.plot(Test_likely, label='Test Likelihood',color = 'green')
    plt.plot(train_avg, label='average Train Likelihood',color = 'black')
    #plt.plot(val_avg, label='average Valid Likelihood',color = 'blue')
    plt.plot(test_avg, label='average Test Likelihood',color = 'blue')
    plt.xlabel('training instance')
    plt.ylabel('Log likelyhood')
    plt.title('SPN likelyhood')
    plt.legend()
    plt.savefig('SPN_'+dataset+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
    plt.close()
    
    plt.figure(figsize = (15,12))
    plt.bar(range(10),flatness, label=' flatness at minima',color = 'red')
    plt.xlabel('training instance')
    plt.ylabel('flatness')
    plt.title('Flatness of minima')
    plt.legend()
    plt.savefig('SPN_flatness_'+ dataset+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
    plt.close()
       
    """
    plt.figure(figsize = (15,12))
    plt.bar(range(10),norm_flatness, label=' norm flatness at minima',color = 'red')
    plt.xlabel('training instance')
    plt.ylabel('flatness')
    plt.title('Flatness of minima')
    plt.legend()
    plt.savefig('SPN_normflatness_'+ dataset+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
    plt.close()
    """
    # Numbers of pairs of bars you want
    N = 10

    # Data on X-axis

    # Specify the values of blue bars (height)
    blue_bar = np.array(Train_likely)*-1
    # Specify the values of orange bars (height)
    orange_bar = np.array(Test_likely)*-1

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
    ax1.set_xlabel('training instance')
    ax1.set_ylabel('- Log likelyhood', color=color)
    # Plotting
    ax1.bar(ind, blue_bar , width, label='Train Likelihood')
    ax1.bar(ind + width, orange_bar, width, label='Test Likelihood')
    ymin = min(min(blue_bar),min(orange_bar))
    ymax = max(max(blue_bar),max(orange_bar))
    ax1.set_ylim([ymin-0.1,ymax+0.004]) 
    ax1.legend(loc='upper left')
    color = 'red'
    ax2 = ax1.twinx()
    ax2.set_ylabel('flatness', color=color)
    ax2.bar(ind+2*width, flatness , width, label='flatness',color=color)
    ax2.legend(loc='upper right')
    plt.title('SPN likelyhood flatness')
    plt.savefig('SPN_ll_flat_bar_'+ dataset+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
    plt.close()
    
    """
    # Figure size
    plt.figure(figsize=(15,12))
    fig, ax1 = plt.subplots()
    fig.set_figheight(12)
    # set width of each subplot as 8
    fig.set_figwidth(15)
    color = 'blue' 
    # Width of a bar 
    width = 0.3       
    ax1.set_xlabel('training instance')
    ax1.set_ylabel('- Log likelyhood', color=color)
    # Plotting
    ax1.bar(ind, blue_bar , width, label='Train Likelihood')
    ax1.bar(ind + width, orange_bar, width, label='Test Likelihood')
    ax1.set_ylim([5.5,6.1]) 
    ax1.legend(loc='best')
    color = 'red'
    ax2 = ax1.twinx()
    ax2.set_ylabel('norm flatness', color=color)
    ax2.bar(ind+2*width, norm_flatness , width, label=' norm flatness',color=color)
    ax2.legend()
    plt.title('SPN likelyhood flatness')
    plt.savefig('SPN_ll_normflat_bar_'+ dataset+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
    plt.close()    
    """
        
    # Figure size
    plt.figure(figsize=(15,12))
    fig, ax1 = plt.subplots()
    fig.set_figheight(12)

    # set width of each subplot as 8
    fig.set_figwidth(15)
    color = 'blue' 
    # Width of a bar 
    width = 0.2     
    ax1.set_xlabel('training instance')
    ax1.set_ylabel('difference in likelyhood', color=color)
    # Plotting
    ax1.bar(ind, abs(blue_bar-orange_bar) , width, label='Likelihood difference')
    ymin = min(abs(blue_bar-orange_bar))
    ymax = max(abs(blue_bar-orange_bar))
    ax1.set_ylim([0,ymax+0.004]) 
    ax1.legend(loc='upper left')
    color = 'red'
    ax2 = ax1.twinx()
    ax2.set_ylabel('flatness', color=color)
    ax2.bar(ind+2*width, flatness , width, label='flatness',color=color)
    ax2.legend(loc='upper right')
    plt.title('SPN likelyhood flatness')
    plt.savefig('SPN_llDiff_flat_bar_'+ dataset+'_'+str(num_latents)+'_'+ str(depth)+'_'+ str(num_repetitions))
    plt.close()
