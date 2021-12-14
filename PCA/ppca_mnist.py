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
from matplotlib.widgets import Slider, Button

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

def reconstruct_1dim(Z):
    recon = scaler.inverse_transform(np.transpose(np.dot(W,Z)))
    rec = np.reshape(recon,(28,28))
    return rec

def update_plot(val):
        z_arr = np.array([z1_slider.val, z2_slider.val, z3_slider.val, z4_slider.val], dtype=np.single)
        plot.set_data(reconstruct_1dim(z_arr))
        fig.canvas.draw_idle()

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
desired_dim = 4


#%% ppca part
# ppca

old_W = np.random.rand(D,desired_dim)
old_sig = 2.1
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

W = old_W
sig = old_sig

print(end-start)



#%% plotting both original and reconstructed datasets

recon = scaler.inverse_transform(np.dot(z,np.transpose(W)))
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

#%% plot the principal components:
pc = scaler.inverse_transform(np.transpose(W))
pc_plottable = np.reshape(pc,(desired_dim,28,28))
fig, axs = plt.subplots(1, 4)

for i in range(desired_dim):
    #plt.subplot(1,2,2)
    plt.imshow(pc_plottable[i], cmap=plt.get_cmap('gray'))
    plt.title("principal components by ppca")
    #plt.show()


        #fig.suptitle(titlestr, fontsize=16)
    axs[i].imshow(pc_plottable[i], cmap = plt.get_cmap('gray'))
        #axs[i].set_title(labelstest[i])
        
        # axs[1].imshow(rec_ica_test_plottable[i], cmap=plt.get_cmap('gray'))
        # axs[1].set_title("ICA recon")

        # axs[2].imshow(rec_ica_test_plottable_comp0[i], cmap=plt.get_cmap('gray'))
        # axs[2].set_title("Comp /= 10")
        # axs[3].imshow(rec_ica_test_plottable_comp1[i], cmap=plt.get_cmap('gray'))
        # axs[3].set_title("Comp *= 10")
plt.show()

#%% Slider:
    
fig = plt.figure()
plot = plt.imshow(reconstruct_1dim(z[1]), cmap='gist_gray')
plt.subplots_adjust(bottom=0.4)
plt.subplots_adjust(left=0.25, bottom=0.25)

z_arr = z[1]

ax_z1 = plt.axes([0.25, 0.1, 0.65, 0.03]) 
z1_slider = Slider(ax=ax_z1, label='Z1', valmin=-1, valmax=1, valinit=z_arr[0])
ax_z2 = plt.axes([0.25, 0.15, 0.65, 0.03])
z2_slider = Slider(ax=ax_z2, label='Z2', valmin=-1, valmax=1.1, valinit=z_arr[1])
ax_z3 = plt.axes([0.25, 0.2, 0.65, 0.03])
z3_slider = Slider(ax=ax_z3, label='Z3', valmin=-1, valmax=1, valinit=z_arr[2])
ax_z4 = plt.axes([0.25, 0.25, 0.65, 0.03])
z4_slider = Slider(ax=ax_z4, label='Z4', valmin=-1.1, valmax=1, valinit=z_arr[3])

z1_slider.on_changed(update_plot)
z2_slider.on_changed(update_plot)
z3_slider.on_changed(update_plot)
z4_slider.on_changed(update_plot)



