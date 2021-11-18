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

mnist = fetch_openml('mnist_784')

# These are the images
mnist.data.shape

# These are the labels
mnist.target.shape

# test_size: what proportion of original data is used for test set
train_img, test_img, train_lbl, test_lbl = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)

scaler = StandardScaler()

# Fit on training set only. Computes the mean and std to be used for later scaling.
scaler.fit(train_img)

# Applies standardization to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

pca = PCA(.95)

pca.fit(train_img)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)


# all parameters not specified are set to their defaults
# default solver is incredibly slow thats why we change it
# solver = 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')

# train your ML
logisticRegr.fit(train_img, train_lbl)

# Returns a NumPy Array
# Predict for One Observation (image)
logisticRegr.predict(test_img[0].reshape(1,-1))

# Predict for Multiple Observations (images) at Once
logisticRegr.predict(test_img[0:10])

score = logisticRegr.score(test_img, test_lbl)
print("accuracy score:",score)

