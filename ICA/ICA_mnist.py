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
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import copy



# Optional way to get mnist, but now it's type is tuple
# from keras.datasets import mnist
# (train_X, train_y), (test_X, test_y) = mnist.load_data()


#%% importing dataset, takes a while...
mnist = fetch_openml('mnist_784')

#%% extra data
matr01 = np.zeros((2,784))
for i in range(1,784):
    matr01[1,i] = 1000
matr01plot = np.reshape(matr01,(2,28,28))
#%% Get data

# These are the images
mnist.data.shape
      
# These are the labels
mnist.target.shape

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
train_img2 = scaler.transform(train_img)
test_img2 = scaler.transform(test_img)

train_img3 = scaler.transform(train_img)
test_img3 = scaler.transform(test_img)

#%% PCA

# Determine how much variance the pc should describe and apply pca:
pca = PCA(.95)
pca.fit(train_img2)
train_imgpca = pca.transform(train_img2)
test_imgpca = pca.transform(test_img2)

# reconstructing the feature space with latent space:
reconpca = pca.inverse_transform(train_imgpca)
reconpca = scaler.inverse_transform(reconpca)
recpca_plottable = np.reshape(reconpca,(60000,28,28))


recon_pca_test = pca.inverse_transform(test_imgpca)
recon_pca_test = scaler.inverse_transform(recon_pca_test)
rec_pca_test_plottable = np.reshape(recon_pca_test,(10000,28,28))


#%%Applying ICA

# # # Give the amount of components wanted and apply ICA:
ica = FastICA(n_components=4,random_state=0, max_iter = 1000)


ica.fit(train_img3)
train_imgica = ica.transform(train_img3)
test_imgica = ica.transform(test_img3)

# train_imgica = ica.fit_transform(train_img3)
# test_imgica = ica.fit_transform(test_img3)

# reconstructing the feature space with latent space:
reconica = ica.inverse_transform(train_imgica)
reconica = scaler.inverse_transform(reconica)
recica_plottable = np.reshape(reconica,(60000,28,28))


recon_ica_test = ica.inverse_transform(test_imgica)
recon_ica_test = scaler.inverse_transform(recon_ica_test)
rec_ica_test_plottable = np.reshape(recon_ica_test,(10000,28,28))


A = ica.mixing_  # Get estimated mixing matrix

#%% 01
matr01ica = ica.transform(matr01)
recon_matr01ica = ica.inverse_transform(matr01ica)
recon_matr01ica = scaler.inverse_transform(recon_matr01ica)
rec_ica_matr01ica_plottable = np.reshape(recon_matr01ica,(2,28,28))

#%% Try find components

print(test_imgica.shape)
print(test_imgica[0])

probeer = copy.copy(test_imgica)
probeer[0,1] = 1
print(probeer[0])

recon_ica_test_probeer = ica.inverse_transform(probeer)
recon_ica_test_probeer = scaler.inverse_transform(recon_ica_test_probeer)
rec_ica_test_plottable_probeer = np.reshape(recon_ica_test_probeer,(10000,28,28))

#%% plotting from train:
print("Plotting Train")
for i in range(10):
     print(i)
     print('PCA:', 1/784*sum(sum(abs(plottable[i]-recpca_plottable[i]))))
     print('ICA:', 1/784*sum(sum(abs(plottable[i]-recica_plottable[i]))))
     plt.subplot(2,2,1)
     plt.title(labels[i])
     plt.imshow(plottable[i], cmap = plt.get_cmap('gray'))
     plt.subplot(2,2,2)
     plt.imshow(recpca_plottable[i], cmap=plt.get_cmap('gray'))
     plt.title("PCA")
     plt.subplot(2,2,3)
     plt.imshow(recica_plottable[i], cmap=plt.get_cmap('gray'))
     plt.title("ICA")
     plt.show()
#%% Plotting from test
print("Plotting Test")
for i in range(1):
    if labelstest[i] == '0':
        print(i)
        print('PCA:', 1/784*sum(sum(abs(plottest[i]-rec_pca_test_plottable[i]))))
        print('ICA:', 1/784*sum(sum(abs(plottest[i]-rec_ica_test_plottable[i]))))
        plt.subplot(2,2,1)
        plt.title(labelstest[i])
        plt.imshow(plottest[i], cmap = plt.get_cmap('gray'))
        plt.subplot(2,2,2)
        plt.imshow(rec_pca_test_plottable[i], cmap=plt.get_cmap('gray'))
        plt.title("PCA")
        plt.subplot(2,2,3)
        plt.imshow(rec_ica_test_plottable[i], cmap=plt.get_cmap('gray'))
        plt.title("ICA")
        plt.show()
     

#%% Trying to find what components do


for j in range(4):
    Comp = np.reshape(A[:,j],(28,28))
    plt.imshow(Comp, cmap = plt.get_cmap('gray'))
    plt.show()
    for i in range(5):
        print(test_imgica[i])

        comp0 = copy.copy(test_imgica)
        comp0[i,j] = 0
        print(comp0[i])

        recon_ica_test_comp0 = ica.inverse_transform(comp0)
        recon_ica_test_comp0 = scaler.inverse_transform(recon_ica_test_comp0)
        rec_ica_test_plottable_comp0 = np.reshape(recon_ica_test_comp0,(10000,28,28))
        
        comp1 = copy.copy(test_imgica)
        comp1[i,j] = 0.01
        print(comp1[i])

        recon_ica_test_comp1 = ica.inverse_transform(comp1)
        recon_ica_test_comp1 = scaler.inverse_transform(recon_ica_test_comp1)
        rec_ica_test_plottable_comp1 = np.reshape(recon_ica_test_comp1,(10000,28,28))    
        
        titlestr = "Changing Component " + str(j+1) + " for number " + str(labelstest[i])
        fig, axs = plt.subplots(1, 4)
        fig.suptitle(titlestr, fontsize=16)
        axs[0].imshow(plottest[i], cmap = plt.get_cmap('gray'))
        axs[0].set_title(labelstest[i])
        
        axs[1].imshow(rec_ica_test_plottable[i], cmap=plt.get_cmap('gray'))
        axs[1].set_title("ICA recon")

        axs[2].imshow(rec_ica_test_plottable_comp0[i], cmap=plt.get_cmap('gray'))
        axs[2].set_title("Comp = 0")
        axs[3].imshow(rec_ica_test_plottable_comp1[i], cmap=plt.get_cmap('gray'))
        axs[3].set_title("Comp = 0.1")
        plt.show()


#%% Components of mixing matrix
for i in range(4):
    Comp = np.reshape(A[:,i],(28,28))
    plt.imshow(Comp, cmap = plt.get_cmap('gray'))
    plt.show()






#%% Plotting from test
print("Plotting 01")
for i in range(2):
    plt.subplot(2,2,1)
    plt.title(i) 
    plt.imshow(matr01plot[i], cmap = plt.get_cmap('gray'))
    plt.subplot(2,2,2)
    plt.imshow(rec_ica_matr01ica_plottable[i], cmap=plt.get_cmap('gray'))
    plt.title("recon")
    plt.show()
    
    





