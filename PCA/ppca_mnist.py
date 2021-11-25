# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 22:17:46 2021

@author: jobre
"""

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

#%% importing dataset, takes a while...
mnist = fetch_openml('mnist_784')

data = mnist.data.to_numpy()

#%% ppca part

scaler = StandardScaler()
scaler.fit(data)

# Applies standardization to both the training set and the test set.
std_data = scaler.transform(data)

# ppca
ppca = PCA(0.95)
ppca.fit(std_data)
reduced_dim = ppca.transform(std_data)

print("dimensionality after pca:",ppca.n_components_)


# reconstructing the feature space with latent space:
recon = ppca.inverse_transform(reduced_dim)
recon = scaler.inverse_transform(recon)

#%% plotting both original and reconstructed datasets
labels = mnist.target.array
plottable = np.reshape(data,(70000,28,28)).astype('uint8')
rec_plottable = np.reshape(recon,(70000,28,28))

for i in range(9):  
    plt.subplot(1,2,1)
    plt.title(labels[i])
    plt.imshow(plottable[i], cmap = plt.get_cmap('gray'))
    plt.subplot(1,2,2)
    plt.imshow(rec_plottable[i], cmap=plt.get_cmap('gray'))
    plt.title("reconstruction")
    plt.show()



