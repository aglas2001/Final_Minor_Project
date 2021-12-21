# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 09:25:00 2021

@author: jobre
"""

from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from sklearn.metrics import mean_squared_error

#%% Slider tool
def reconstruct(latent_var):
    rec = scaler.inverse_transform(pca.inverse_transform(latent_var))
    rec = np.reshape(rec,(28,28))
    return rec;
    

def update_plot(val):
        z_arr = np.array([z1_slider.val, z2_slider.val, z3_slider.val, z4_slider.val], dtype=np.single)
        plot.set_data(reconstruct(z_arr))
        fig.canvas.draw_idle()


#%% importing dataset, takes a while...
#mnist_dataset = fetch_openml('mnist_784')

from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = np.reshape(train_X,(train_X.shape[0],784))
test_X = np.reshape(test_X,(test_X.shape[0],784))

scaler = StandardScaler()

scaler.fit(train_X)

#standardize data
train_data = scaler.transform(train_X)
test_data = scaler.transform(test_X)

N, D = train_data.shape
mu = np.mean(train_data,axis=0)
desired_dim = 4

#%%

#preserve orginial values test_X for plot later:
plottable = np.reshape(test_X,(test_X.shape[0],28,28))
labels = test_y


# Determine how much variance the pc should describe and apply pca:
pca = PCA(n_components=desired_dim)
pca.fit(train_data)
latent = pca.transform(test_data)

print("dimensionality in latent space:", latent.shape)

# reconstructing the feature space with latent space:
recon = scaler.inverse_transform(pca.inverse_transform(latent))
rec_plottable = np.reshape(recon,(recon.shape[0],28,28))

#%% plotting the principal components

pc = pca.components_
pc_plottable = np.reshape(pc,(desired_dim,28,28))

fig, axs = plt.subplots(1,desired_dim)

for i in range(desired_dim):
    axs[i].imshow(pc_plottable[i], cmap=plt.get_cmap('gray'))

plt.show()
    


#%% plotting a few examples:
for i in range(23,43):  
    plt.subplot(1,2,1)
    plt.title(labels[i])
    plt.imshow(plottable[i], cmap = plt.get_cmap('gray'))
    plt.subplot(1,2,2)
    plt.imshow(rec_plottable[i], cmap=plt.get_cmap('gray'))
    plt.title("reconstruction")
    plt.show()

#%% manual reconstruction
# man_rec = np.dot(latent,pc)
# rec = scaler.inverse_transform(man_rec)
# man_rec_plottable = np.reshape(rec,(N,28,28))

# for i in range(9):  
#     plt.subplot(1,2,1)
#     plt.title("original reconstruction")
#     plt.imshow(rec_plottable[i], cmap = plt.get_cmap('gray'))
#     plt.subplot(1,2,2)
#     plt.imshow(man_rec_plottable[i], cmap=plt.get_cmap('gray'))
#     plt.title("manual reconstruction")
#     plt.show()

       
        
#%%

fig = plt.figure()
plot = plt.imshow(reconstruct(latent[0]), cmap='gist_gray')
plt.subplots_adjust(bottom=0.4)
plt.subplots_adjust(left=0.25, bottom=0.25)

z_arr = latent[0]

ax_z1 = plt.axes([0.25, 0.1, 0.65, 0.03]) 
z1_slider = Slider(ax=ax_z1, label='Z1', valmin=-20, valmax=20, valinit=z_arr[0])
ax_z2 = plt.axes([0.25, 0.15, 0.65, 0.03])
z2_slider = Slider(ax=ax_z2, label='Z2', valmin=-20, valmax=20, valinit=z_arr[1])
ax_z3 = plt.axes([0.25, 0.2, 0.65, 0.03])
z3_slider = Slider(ax=ax_z3, label='Z3', valmin=-20, valmax=20, valinit=z_arr[2])
ax_z4 = plt.axes([0.25, 0.25, 0.65, 0.03])
z4_slider = Slider(ax=ax_z4, label='Z4', valmin=-20, valmax=20, valinit=z_arr[3])

z1_slider.on_changed(update_plot)
z2_slider.on_changed(update_plot)
z3_slider.on_changed(update_plot)
z4_slider.on_changed(update_plot)


#%% see influence of latent space:
    
fig, axs = plt.subplots(1,4)
z = latent[0]
add= np.array([0,-7,0,0])
for i in range(desired_dim):
    axs[i].imshow(reconstruct(z + i * add),cmap='gist_gray')
    

plt.show()

#%% plot reconstruction error for different latent dimensionalities

# Determine how much variance the pc should describe and apply pca:

    
mse = np.zeros(13)
for i in range(13):
    pca = PCA(n_components=i+1)
    pca.fit(train_data)
    lat = pca.transform(test_data)
    rec = scaler.inverse_transform(pca.inverse_transform(lat))
    mse[i] = mean_squared_error(test_X,rec)
    
plt.plot(mse)
plt.xlabel("Dimensionality of latent space")
plt.ylabel("Reconstruction error (MSE)")
plt.xlim([1,12])
    

