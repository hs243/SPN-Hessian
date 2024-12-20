
"""
Train a PC
==========

This tutorial demonstrates how to create a Hidden Chow-Liu Tree (https://arxiv.org/pdf/2106.02264.pdf) using :code:`pyjuice.structures` and train the model with mini-batch EM and full-batch EM.
For simplicity, we use the MNIST dataset as an example. 

Note that the goal of this tutorial is just to quickly demonstrate the basic training pipeline using PyJuice without covering additional details such as ways to construct a PC, which will be covered in the following tutorials.
"""

# sphinx_gallery_thumbnail_path = 'imgs/juice.png'

# %%
# Load the MNIST Dataset
# ----------------------

import pyjuice as juice
import pyjuice.visualize as juice_vis
import torch
import torchvision
import time
from torch.utils.data import TensorDataset, DataLoader
import pyjuice.nodes.distributions as dists
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import pandas as pd

matplotlib.rcParams.update({'font.size': 30})
train_dataset = torchvision.datasets.MNIST(root = "../data", train = True, download = True)
valid_dataset = torchvision.datasets.MNIST(root = "../data", train = False, download = True)

Train_size = [5000,10000,20000,40000,60000]
Valid_data = valid_dataset.data

LIKELIHOOD = []
Bits_per_dim = []
Deg_over = []


Latents = [16,32,64,128]
str_lat = ['16','32','64','128']
# %%
# Let's create a HCLT PC with latent size 128.
for num_latents in Latents:
    device = torch.device("cuda:0")
    depth = 10
    num_repetitions = 1
    BPD = []
    Overfitting = []
    for size in Train_size:
        Train_data = train_dataset.data[0:size]

        train_data = Train_data.reshape(size, 28*28)
        valid_data = Valid_data.reshape(len(valid_dataset.data), 28*28)

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


   
        TRAIN_like = np.zeros(2000)
        TEST_like = np.zeros(2000)
        for trial in range(1):

            # The data is required to construct the backbone Chow-Liu Tree structure for the HCLT
            ns = juice.structures.RAT_SPN(num_vars = len(train_data[0]), 
                    num_latents = num_latents,
                    depth = depth,
                    num_repetitions = num_repetitions,
                    input_node_params= {"num_cats": 256}
                )        


            # %%
            # :code:`ns` is a Directed Acyclic Graph (DAG) representation of the PC. 
            # Specifically, we use :code:`pyjuice.nodes.InputNodes`, :code:`pyjuice.nodes.ProdNodes`, and :code:`pyjuice.nodes.SumNodes` to define vectors of input nodes, product nodes, and sum nodes,   respectively.
            # By also storing the topological structure of the node vectors (with pointers to the child node vectors), we create the PC as a DAG-based structure. :code:`ns` is also just a node vector defining the root node of the PC.
            #    
            # While being user-friendly, the DAG-based representation is not amenable to efficient computation. 
            # Therefore, before doing any computation, we need to compile the PC with :code:`pyjuice.compile`, which creates a compact and equivalent representation of the PC.

            pc = juice.compile(ns)  

            # %%
            # The :code:`pc` is an instance of :code:`torch.nn.Module`. So we can safely assume it is just a neural network with the variable assignments :math:`\mathbf{x}` as input and its log-    likelihood :math:`\log p(\mathbf{x})` as output. 
            # We proceed to move it to the GPU specified by :code:`device`.

            pc.to(device)
            plt.figure()
            juice_vis.plot_pc(ns, node_id = False, node_num_label = False)
            plt.savefig('SPN_MNIST_28_256_'+str(depth))
            plt.close()
            # %%
            # Train the PC
            # ------------

            # %%
            # We start by defining the optimizer and scheduler.
 
            optimizer = juice.optim.CircuitOptimizer(pc, lr = 0.1, pseudocount = 0.1, method = "EM")
            scheduler = juice.optim.CircuitScheduler(
                optimizer, 
                method = "multi_linear", 
                lrs = [0.99, 0.9, 0.05], 
                milestone_steps = [0, len(train_loader) * 100, len(train_loader) * 350]
            )

            # %%
            # Optionally, we can leverage CUDA Graphs to hide the kernel launching overhead by doing a dry run.

            for batch in train_loader:
                x = batch[0].to(device)
                print(x.shape)
                lls = pc(x, record_cudagraph = True)
                lls.mean().backward()
                break

            training_ll = lls.mean().detach().cpu().numpy().item()                            
            print('Train LL = ',training_ll) 
            # %%
            # We are now ready for the training. Below is an example training loop for mini-batch EM.
            Train_ll = []
            Test_ll =[]
            for epoch in range(1, 2000+1):
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
                #print(f"[Epoch {epoch}/{350}][train LL: {train_ll:.2f}; val LL: {test_ll:.2f}].....[train forward+backward+step {t1-t0:.2f}; val forward {t2-t1:.2f}] ")

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
        plt.title('Difference in likelihood' + '_'+str(num_latents))
        plt.legend()
        plt.savefig(' likelihood_28_256_'+str(num_latents)+'_'+str(depth)+'_'+str(size))
        plt.close()     
        
        deg_over = (TEST_like[-1] - TRAIN_like[-1])/TEST_like[-1]
        Overfitting.append(deg_over)
        bpd = (-(TEST_like/1)[-1])/(np.log(2)*28*28)
        BPD.append(bpd)
    Bits_per_dim.append(BPD)
    Deg_over.append(Overfitting)

plt.figure(figsize=(25,20))
plt.xlabel('Training size')
plt.ylabel('BPD')
for i in range(len(Bits_per_dim)):
    plt.plot(Train_size,Bits_per_dim[i], marker = 'o', label='Latent size ' + str_lat[i])
plt.title('Bits_per_dim vs Training size' )
plt.legend()
plt.savefig('Bits_per_dim')      
plt.close()

plt.figure(figsize=(25,20))
plt.xlabel('Training size')
plt.ylabel('BPD')
for i in range(len(Deg_over)):
    plt.plot(Train_size,Deg_over[i], marker = 'o', label='Latent size ' + str_lat[i])
plt.title('Degree of overfitting vs Training size' )
plt.legend()
plt.savefig('Degree of overfitting') 
plt.close()
