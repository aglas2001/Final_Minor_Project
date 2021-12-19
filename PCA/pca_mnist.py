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
from matplotlib.widgets import Slider, Button

#%% Slider tool
def reconstruct(latent_var):
    rec = scaler.inverse_transform(pca.inverse_transform(latent_var))
    rec = np.reshape(rec,(28,28))
    return rec;
    

def update_plot(val):
        z_arr = np.array([z1_slider.val, z2_slider.val, z3_slider.val, z4_slider.val], dtype=np.single)
        plot.set_data(reconstruct(z_arr))
        fig.canvas.draw_idle()

# Optional way to get mnist, but now it's type is tuple
# from keras.datasets import mnist
# (train_X, train_y), (test_X, test_y) = mnist.load_data()


#%% importing dataset, takes a while...
mnist_dataset = fetch_openml('mnist_784')

#%% prepare data
mnist = mnist_dataset.data.to_numpy()

scaler = StandardScaler()

scaler.fit(mnist)

#standardize data
data = scaler.transform(mnist)

N, D = data.shape
mu = np.mean(data,axis=0)
desired_dim = 4
#%%

#preserve orginial values train_img for plot later:
plottable = np.reshape(mnist,(N,28,28))
labels = mnist_dataset.target.array


# Determine how much variance the pc should describe and apply pca:
pca = PCA(n_components=desired_dim)
pca.fit(data)
latent = pca.transform(data)

print("dimensionality in latent space:", latent.shape)

# reconstructing the feature space with latent space:
recon = pca.inverse_transform(latent)
recon = scaler.inverse_transform(recon)
rec_plottable = np.reshape(recon,(N,28,28))

#%% plotting the principal components
pc = pca.components_
pc_plottable = np.reshape(pc,(desired_dim,28,28))

for i in range(desired_dim):
    plt.subplot(1,2,2)
    plt.imshow(pc_plottable[i], cmap=plt.get_cmap('gray'))
    plt.title("principal component ")
    plt.show()
    


#%% plotting a few examples:
for i in range(9):  
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
z = latent[1]
add= np.array([0,0,0,1])
for i in range(desired_dim):
    z = z + i * add
    axs[i].imshow(reconstruct(z),cmap='gist_gray')
    

plt.show()




