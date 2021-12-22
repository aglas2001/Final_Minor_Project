# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:22:09 2021

@author: aglas
"""

from sklearn.datasets import fetch_openml
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import math as m
from sklearn.decomposition import PCA
import copy



def loadmnist():
    mnist_dataset = fetch_openml('mnist_784')
    return mnist_dataset


def loaddata(mnist_dataset):
    mnist = mnist_dataset.data.to_numpy()
          
    scaler = StandardScaler()
    scaler.fit(mnist)
    data = scaler.transform(mnist)
    N, D = data.shape
    mu = np.mean(data,axis=0)
    
    plottable = np.reshape(mnist,(N,28,28))
    labels = mnist_dataset.target.array
    return data, scaler, N, D, mu,labels,plottable

def loaddatasplit(mnist_dataset):
    mnist = mnist_dataset.data.to_numpy()

    # test_size: what proportion of original data is used for test set
    train_img, test_img, train_lbl, test_lbl = train_test_split(
        mnist.data, mnist.target, test_size=1/7.0, random_state=0)

    #preserve orginial values train_img for plot later:
    plottable = np.reshape(train_img.to_numpy(),(60000,28,28))
    labels = train_lbl.array

    plottest = np.reshape(test_img.to_numpy(),(10000,28,28))
    labelstest = test_lbl.array

    # Fit on training set only. Computes the mean and std to be used for later scaling.
    scaler = StandardScaler()
    scaler.fit(train_img)

    # Applies standardization to both the training set and the test set.
    return train_img,test_img,plottable,labels,plottest,labelstest,scaler





def ApplyPCA(desired_dim,scaler, data,N):
    pca = PCA(n_components=desired_dim)
    pca.fit(data)
    train_imgpca = pca.transform(data)
    
    # reconstructing the feature space with latent space:
    reconpca = pca.inverse_transform(train_imgpca)
    reconpca = scaler.inverse_transform(reconpca)
    recpca_plottable = np.reshape(reconpca,(N,28,28))
    
    return recpca_plottable



def ApplyICA(desired_dim,scaler, data,N):
    ica = FastICA(n_components=4,random_state=0, max_iter = 1000)
    
    
    ica.fit(data)
    train_imgica = ica.transform(data)
        
    reconica = ica.inverse_transform(train_imgica)
    reconica = scaler.inverse_transform(reconica)
    recica_plottable = np.reshape(reconica,(N,28,28))
        
    A = ica.mixing_  # Get estimated mixing matrix
    return ica,train_imgica, recica_plottable, A
    

def ReconstructionError(plottable,recpca_plottable,recica_plottable,labels):
    rmsePCA = m.sqrt(mean_squared_error(plottable, recpca_plottable))
    nrmsePCA = rmsePCA/m.sqrt(np.mean(plottable**2))
    rmseICA = m.sqrt(mean_squared_error(plottable, recica_plottable))
    nrmseICA = rmseICA/m.sqrt(np.mean(plottable**2))
    #errorstringPCA = "Reconstruction Error for " + str(labels[i])+ " using PCA: "+ str(nrmsePCA)
    #errorstringICA = "Reconstruction Error for " + str(labels[i])+ " using ICA: "+ str(nrmseICA)
    return nrmsePCA,nrmseICA


def PlotReconstruction(AmountOfPlots, plottable,recpca_plottable,recica_plottable,labels):
    for i in range(AmountOfPlots):
        rmsePCA = m.sqrt(mean_squared_error(plottable[i], recpca_plottable[i]))
        nrmsePCA = rmsePCA/m.sqrt(np.mean(plottable**2))
        rmseICA = m.sqrt(mean_squared_error(plottable[i], recica_plottable[i]))
        nrmseICA = rmseICA/m.sqrt(np.mean(plottable**2))

        errorstringPCA = "Reconstruction Error for " + str(labels[i])+ " using PCA: "+ str(nrmsePCA)
        errorstringICA = "Reconstruction Error for " + str(labels[i])+ " using ICA: "+ str(nrmseICA)
        print(errorstringPCA)
        print(errorstringICA)
        titlestr = "Reconstructing " + str(labels[i])+ " using ICA"
        fig, axs = plt.subplots(1, 3)
        fig.suptitle(titlestr, fontsize=16)
        axs[0].imshow(plottable[i], cmap = plt.get_cmap('gray'))
        axs[0].set_title(labels[i])
        
        axs[1].imshow(recpca_plottable[i], cmap=plt.get_cmap('gray'))
        axs[1].set_title("PCA recon")
        
        axs[2].imshow(recica_plottable[i], cmap=plt.get_cmap('gray'))
        axs[2].set_title("ICA recon")
     


def Plotcomponents(AmountOfPlots, AmountOfComponents, N, ica, A, scaler, train_imgica, labels, plottable,recica_plottable):
    for j in range(AmountOfComponents):
        Comp = np.reshape(A[:,j],(28,28))
        plt.imshow(Comp, cmap = plt.get_cmap('gray'))
        plt.show()
        for i in range(AmountOfPlots):
    
            comp0 = copy.copy(train_imgica)
            comp0[i,j] /= 10
    
            recon_ica_test_comp0 = ica.inverse_transform(comp0)
            recon_ica_test_comp0 = scaler.inverse_transform(recon_ica_test_comp0)
            rec_ica_test_plottable_comp0 = np.reshape(recon_ica_test_comp0,(N,28,28))
            
            comp1 = copy.copy(train_imgica)
            comp1[i,j] *= 10
    
            recon_ica_test_comp1 = ica.inverse_transform(comp1)
            recon_ica_test_comp1 = scaler.inverse_transform(recon_ica_test_comp1)
            rec_ica_test_plottable_comp1 = np.reshape(recon_ica_test_comp1,(N,28,28))    
            
            titlestr = "Changing Component " + str(j+1) + " for number " + str(labels[i])
            fig, axs = plt.subplots(1, 4)
            fig.suptitle(titlestr, fontsize=16)
            axs[0].imshow(plottable[i], cmap = plt.get_cmap('gray'))
            axs[0].set_title(labels[i])
            
            axs[1].imshow(recica_plottable[i], cmap=plt.get_cmap('gray'))
            axs[1].set_title("ICA recon")
    
            axs[2].imshow(rec_ica_test_plottable_comp0[i], cmap=plt.get_cmap('gray'))
            axs[2].set_title("Comp /= 10")
            axs[3].imshow(rec_ica_test_plottable_comp1[i], cmap=plt.get_cmap('gray'))
            axs[3].set_title("Comp *= 10")
            plt.show()


def ShowComponents(AmountOfComponents, A):
    for i in range(AmountOfComponents):
        Comp = np.reshape(A[:,i],(28,28))
        plt.imshow(Comp, cmap = plt.get_cmap('gray'))
        plt.show()






desired_dim=4


#%%
mnist_dataset = loadmnist()
#%%
data, scaler, N, D, mu,labels,plottable = loaddata(mnist_dataset)
#%%
train_img,test_img,plottable,labels,plottest,labelstest,scaler = loaddatasplit(mnist_dataset)
#%%
recpca_plottable = ApplyPCA(desired_dim,scaler, data,N)
#%%
ica,train_imgica, recica_plottable, A = ApplyICA(desired_dim,scaler, data,N)
#%%
PlotReconstruction(1, plottable,recpca_plottable,recica_plottable,labels)
#%%
Plotcomponents(1, desired_dim,N, ica, A, scaler, train_imgica, labels, plottable,recica_plottable)

#%%
nrmsePCA,nrmseICA = ReconstructionError(plottable,recpca_plottable,recica_plottable,labels)






#%%
def reconstruct(latentspace):
    recon_ica_test_comp0 = ica.inverse_transform([latentspace])
    recon_ica_test_comp0 = scaler.inverse_transform(recon_ica_test_comp0)
    rec_ica_test_plottable_comp0 = np.reshape(recon_ica_test_comp0,(28,28))
    return rec_ica_test_plottable_comp0


def update(val):
    z_arr = np.array([z1_slider.val, z2_slider.val, z3_slider.val, z4_slider.val], dtype=np.single)
    plot.set_data(reconstruct(z_arr))
    fig.canvas.draw_idle()


#%% Slider
I = 0


fig = plt.figure()
plot = plt.imshow(recica_plottable[I], cmap=plt.get_cmap('gray'))
plt.subplots_adjust(bottom=0.4)

z_arr = np.zeros(4)

ax_z1 = plt.axes([0.25, 0.1, 0.65, 0.03])
z1_slider = Slider(ax=ax_z1, label='Z1', valmin=-0.02, valmax=0.02, valinit=z_arr[0])
ax_z2 = plt.axes([0.25, 0.15, 0.65, 0.03])
z2_slider = Slider(ax=ax_z2, label='Z2', valmin=-0.02, valmax=0.02, valinit=z_arr[1])
ax_z3 = plt.axes([0.25, 0.2, 0.65, 0.03])
z3_slider = Slider(ax=ax_z3, label='Z3', valmin=-0.02, valmax=0.02, valinit=z_arr[2])
ax_z4 = plt.axes([0.25, 0.25, 0.65, 0.03])
z4_slider = Slider(ax=ax_z4, label='Z4', valmin=-0.02, valmax=0.02, valinit=z_arr[3])
    
z1_slider.on_changed(update)
z2_slider.on_changed(update)
z3_slider.on_changed(update)
z4_slider.on_changed(update)