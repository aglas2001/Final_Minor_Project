# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 22:17:46 2021

@author: jobre
"""

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sp
import scipy.sparse.linalg as la
import pandas as pd
import time

def get_exp_z(W,sigma,M):
    exp_z = np.zeros((N,desired_dim))

    
    for i in range(0,N):
        exp_z[i,:] = np.dot( (linalg.inv(M) @ np.transpose(W)) , (data[i,:]-mu) ) #may need to transpose this # changed from np.dot(linalg.inv(M),np.transpose(W))
        
    return exp_z

def get_exp_zzt(exp_z,M,sig):
    return sig**2 * linalg.inv(M) + np.outer(exp_z,exp_z) #changed from np.dot(exp_z,np.transpose(exp_z))

def E_step(W,sigma):
    M = (np.transpose(W) @ W) + sigma**2*sp.eye(desired_dim) #changed from np.dot(np.transpose(W),W)
    
    exp_z = get_exp_z(W,sigma,M)
    
    return exp_z,M
    

def get_W(exp_z,M,prev_sig):
    W_one = np.zeros((D,desired_dim))
    W_two = np.zeros((desired_dim,desired_dim))
    
    for i in range(0,N):
        W_one += np.outer((data[i,:]-mu),exp_z[i,:])
        W_two += get_exp_zzt(exp_z[i,:],M,prev_sig)
        
    W_new = W_one @ linalg.inv(W_two)
    
    return W_new
    

def get_sigma(W,exp_z,M,prev_sig):
    sig_new = 0
    
    for i in range(0,N):
        exp_zzt = get_exp_zzt(exp_z[i,:], M, prev_sig)
        sig_new += linalg.norm(data[i,:] - mu)**2 - 2*np.dot(np.dot(exp_z[i,:],np.transpose(W)),(data[i,:]-mu)) + np.trace( exp_zzt @ np.transpose(W) @ W) 

    sig_new = np.sqrt((1/(N*D))*sig_new) # no sqrt()!!
    return sig_new


def M_step(exp_z,M,prev_sig):
    W_new = get_W(exp_z,M,prev_sig)
    sig_new = get_sigma(W_new,exp_z,M,prev_sig)
    return W_new,sig_new

#%% importing dataset, takes a while...
mnist_dataset = fetch_openml('mnist_784')

#%% prepare data
MNIST = mnist_dataset.data.to_numpy()

scaler = StandardScaler()

scaler.fit(MNIST)

#standardize data
data = scaler.transform(MNIST)

N, D = data.shape
mu = np.mean(data,axis=0)
desired_dim = 10


#%% ppca part
# ppca

old_W = np.random.rand(D,desired_dim)
old_sig = 5
k = 0
conv = 1000

start = time.time()
while conv > 1e-4:
    z,temp_M = E_step(old_W, old_sig)
    old_W, new_sig = M_step(z,temp_M,old_sig)
    conv = abs(old_sig - new_sig)
    old_sig = new_sig
    k+=1
    print(k)

end = time.time()

print(end-start)



#%% plotting both original and reconstructed datasets

recon = scaler.inverse_transform(np.transpose(np.dot(old_W,np.transpose(z))))
labels = mnist_dataset.target.array
plottable = np.reshape(MNIST,(70000,28,28)).astype('uint8')
rec_plottable = np.reshape(recon,(70000,28,28))

for i in range(9):  
    plt.subplot(1,2,1)
    plt.title(labels[i])
    plt.imshow(plottable[i], cmap = plt.get_cmap('gray'))
    plt.subplot(1,2,2)
    plt.imshow(rec_plottable[i], cmap=plt.get_cmap('gray'))
    plt.title("reconstruction")
    plt.show()



