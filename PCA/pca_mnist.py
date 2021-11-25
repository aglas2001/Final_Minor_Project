# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:25:00 2021

@author: jobre
"""

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

# Optional way to get mnist, but now it's type is tuple
# from keras.datasets import mnist
# (train_X, train_y), (test_X, test_y) = mnist.load_data()


#%% importing dataset, takes a while...
mnist = fetch_openml('mnist_784')


#%% processing data, applying pca and plotting original and reconstructed values

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

# Fit on training set only. Computes the mean and std to be used for later scaling.
scaler = StandardScaler()
scaler.fit(train_img)

# Applies standardization to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

# Determine how much variance the pc should describe and apply pca:
pca = PCA(.95)
pca.fit(train_img)
train_img = pca.transform(train_img)
test_img = pca.transform(test_img)
print("dimensionality after pca:",train_img.shape)

# reconstructing the feature space with latent space:
recon = pca.inverse_transform(train_img)
recon = scaler.inverse_transform(recon)
rec_plottable = np.reshape(recon,(60000,28,28))

# plotting a few examples:
for i in range(9):  
    plt.subplot(1,2,1)
    plt.title(labels[i])
    plt.imshow(plottable[i], cmap = plt.get_cmap('gray'))
    plt.subplot(1,2,2)
    plt.imshow(rec_plottable[i], cmap=plt.get_cmap('gray'))
    plt.title("reconstruction")
    plt.show()



#%% potential machine learning:
# # all parameters not specified are set to their defaults
# # default solver is incredibly slow thats why we change it
# # solver = 'lbfgs'
# logisticRegr = LogisticRegression(solver = 'lbfgs')

# # train your ML
# logisticRegr.fit(train_img, train_lbl)

# # Returns a NumPy Array
# # Predict for One Observation (image)
# logisticRegr.predict(test_img[0].reshape(1,-1))

# # Predict for Multiple Observations (images) at Once
# logisticRegr.predict(test_img[0:10])

# score = logisticRegr.score(test_img, test_lbl)
# print("accuracy score:",score)

