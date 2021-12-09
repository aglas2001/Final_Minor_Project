# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 08:14:00 2021

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
    

#%% prepare data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# load dataset into Pandas DataFrame
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])

features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
data = StandardScaler().fit_transform(x)
N, D = data.shape
mu = np.mean(data,axis=0)
desired_dim = 2

#%% EM test

old_W = np.random.rand(D,desired_dim)
old_sig = 1
k = 0

while k < 1000:
    z,temp_M = E_step(old_W, old_sig)
    old_W, old_sig = M_step(z,temp_M,old_sig)
    print(old_sig)
    k +=1
    

#%% regular PCA to compare:

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])


finalDf = pd.concat([principalDf, df[['target']]], axis = 1)

# plot the 2 principal components
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

#color data to know its true 'value'
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()


#%% Plotting PPCA:
    
principalDf2 = pd.DataFrame(data = z
             , columns = ['principal component 1', 'principal component 2'])


finalDf2 = pd.concat([principalDf2, df[['target']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('Probabilistic PCA', fontsize = 20)

#color data to know its true 'value'
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf2['target'] == target
    ax.scatter(finalDf2.loc[indicesToKeep, 'principal component 1']
               , finalDf2.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

#%% Ignore this
# cov = np.cov(data,rowvar=False)
# eigenValues, eigenVectors = linalg.eig(cov)

# idx = eigenValues.argsort()[::-1]   
# eigenValues = eigenValues[idx]
# eigenVectors = eigenVectors[:,idx]
# print(eigenValues)
    
#%%
# def update_W_sigma(exp_z,old_W,old_sig):
#     W_one = np.zeros((D,desired_dim))
#     W_two = np.zeros((desired_dim,desired_dim))
#     M = np.dot(np.transpose(old_W),old_W) + old_sig*old_sig*sp.eye(desired_dim)

#     for i in range(0,N):
#         W_one += np.outer((data[i,:]-mu),exp_z[i,:])
#         W_two += old_sig*old_sig*linalg.inv(M)+exp_z[i,:]*np.transpose(exp_z[i,:])
        
#     W_new = np.dot(W_one,linalg.inv(W_two))
    
#     sig_new = 0
    
#     for i in range(0,N):
#         exp_zzt = get_exp_zzt(exp_z[i,:],M,old_sig)
#         sig_new += linalg.norm(data[i,:] - mu)**2 - 2*np.dot(np.dot(np.transpose(exp_z[i,:]),np.transpose(W_new)),(data[i,:]-mu)) + np.trace( np.dot( exp_zzt , np.dot(np.transpose(W_new),W_new)) )
    
#     sig_new = int((1/(N*D)) * sig_new)
    
#     return W_new,sig_new

