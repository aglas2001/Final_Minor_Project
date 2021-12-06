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



# Optional way to get mnist, but now it's type is tuple
# from keras.datasets import mnist
# (train_X, train_y), (test_X, test_y) = mnist.load_data()


#%% importing dataset, takes a while...
mnist = fetch_openml('mnist_784')

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

# Fit on training set only. Computes the mean and std to be used for later scaling.
scaler = StandardScaler()
scaler.fit(train_img)

# Applies standardization to both the training set and the test set.
train_img2 = scaler.transform(train_img)
test_img2 = scaler.transform(test_img)

train_img3 = scaler.transform(train_img)
test_img3 = scaler.transform(test_img)

print('Starting Dimensionality:', train_img.shape)

#%% PCA

# Determine how much variance the pc should describe and apply pca:
pca = PCA(.95)
pca.fit(train_img2)
train_imgpca = pca.transform(train_img2)
test_imgpca = pca.transform(test_img2)
print("dimensionality after pca:",train_imgpca.shape)

# reconstructing the feature space with latent space:
reconpca = pca.inverse_transform(train_imgpca)
reconpca = scaler.inverse_transform(reconpca)
recpca_plottable = np.reshape(reconpca,(60000,28,28))

#%% Other PCA
#data = mnist.data.to_numpy()


#scaler = StandardScaler()
#scaler.fit(data)

# Applies standardization to both the training set and the test set.
#std_data = scaler.transform(data)
otherpca = PCA(0.95)
otherpca.fit(train_img2)
reduced_dim = otherpca.transform(train_img2)

print("dimensionality after other pca:",otherpca.n_components_)


# reconstructing the feature space with latent space:
recon = otherpca.inverse_transform(reduced_dim)
recon = scaler.inverse_transform(recon)

labelsppca = mnist.target.array
plottable2 = np.reshape(train_img2,(60000,28,28)).astype('uint8')
rec2_plottable = np.reshape(recon,(60000,28,28))


#%%Applying ICA

# # # Give the amount of components wanted and apply ICA:
ica = FastICA(n_components=327,random_state=0, max_iter = 200)
ica.fit(train_img3)
train_imgica = ica.transform(train_img3)
test_imgica = ica.transform(test_img3)


#train_transformed = ica.fit_transform(train_img3)
#test_transformed = ica.fit_transform(test_img3)

#print("Dimensionality after ICA:",train_imgica.shape)

# # reconstructing the feature space with latent space:
reconica = ica.inverse_transform(train_imgica)
reconica = scaler.inverse_transform(reconica)
recica_plottable = np.reshape(reconica,(60000,28,28))

# recontransform = ica.inverse_transform(train_transformed)
# recontransform = scaler.inverse_transform(recontransform)
# rectransform_plottable = np.reshape(recontransform,(60000,28,28))


#%% plotting a few examples:
for i in range(5):
     print(i)
     print(recpca_plottable[i].shape)
     print(recica_plottable[i].shape)
     print('PCA:', sum(sum(abs(plottable[i]-recpca_plottable[i]))))
     print('ICA:', sum(sum(abs(plottable[i]-recica_plottable[i]))))
#     print('PPCA:', sum(sum(abs(plottable[i]-rec2_plottable[i]))))
     print('PCA-ICA:', sum(sum(abs(recica_plottable[i]-recpca_plottable[i]))))
     plt.subplot(2,2,1)
     plt.title(labels[i])
     plt.imshow(plottable[i], cmap = plt.get_cmap('gray'))
     plt.subplot(2,2,2)
     plt.imshow(recpca_plottable[i], cmap=plt.get_cmap('gray'))
     plt.title("PCA")
     plt.subplot(2,2,3)
     plt.imshow(recica_plottable[i], cmap=plt.get_cmap('gray'))
     plt.title("ICA")
#     plt.subplot(2,2,4)
#     plt.imshow(rec2_plottable[i], cmap=plt.get_cmap('gray'))
#     plt.title("PPCA")
     plt.show()
    #%%
print(labels)