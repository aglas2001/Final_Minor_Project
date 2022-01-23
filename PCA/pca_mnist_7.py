# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:57:27 2021

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

def apply_pca(train, test, desired_dim):
    
    scaler = StandardScaler()
    scaler.fit(train)
    
    #standardize data
    std_data = scaler.transform(train)
    std_test = scaler.transform(test)
    
    pca = PCA(n_components=desired_dim)
    pca.fit(std_data)
    latent = pca.transform(std_test)
    pc = scaler.inverse_transform(pca.components_)
    rec_data = scaler.inverse_transform(pca.inverse_transform(latent))
    
    return pc, latent, rec_data

def MeanRelativeError(test,rec):
    M,N = test.shape
    a = 0
    for i in range(M):
        for j in range(N):
            if abs(test[i][j]) > 10e-10:
                a += abs((test[i][j] - rec[i][j])/test[i][j])
    return a/(M*N) * 100


#%%
# Optional way to get mnist, but now it's type is tuple
from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#%%
#Loading DATA
# from torchvision import datasets
# import torchvision.transforms as transforms
# transform = transforms.Compose([transforms.ToTensor(),])
# training_data = datasets.MNIST(
#     root="data_train",
#     train=True,
#     download=True,
#     transform=transform,
# )

# # Download test data from open datasets.
# validation_data = datasets.MNIST(
#     root="data_val",
#     train=False,
#     download=True,
#     transform=transform,
# )

# train_X = training_data[0][0].squeeze().numpy()

#%%

train_filter = np.where(train_y == 7)
test_filter = np.where(test_y == 7)

train_X, train_y = train_X[train_filter], train_y[train_filter]
test_X, test_y = test_X[test_filter], test_y[test_filter]


train_X = np.reshape(train_X,(train_X.shape[0],784))
test_X = np.reshape(test_X,(test_X.shape[0],784))

scaler = StandardScaler()

scaler.fit(train_X)

# #standardize data
data = scaler.transform(train_X)
test_data = scaler.transform(test_X)

N, D = data.shape
mu = np.mean(data,axis=0)
desired_dim = 4


#%%

#preserve orginial values test_X for plot later:
plottable = np.reshape(test_X,(test_X.shape[0],28,28))
labels = test_y


# Determine how much variance the pc should describe and apply pca:
pca = PCA(n_components=desired_dim)
pca.fit(data)
latent = pca.transform(test_data)

print("dimensionality in latent space:", latent.shape)

# reconstructing the feature space with latent space:
recon = pca.inverse_transform(latent)
recon = scaler.inverse_transform(recon)
rec_plottable = np.reshape(recon,(test_X.shape[0],28,28))

#%% plotting the principal components
pc = pca.components_
pc_plottable = np.reshape(pc,(desired_dim,28,28))

fig, axs = plt.subplots(1,desired_dim)

for i in range(desired_dim):
    axs[i].imshow(pc_plottable[i], cmap=plt.get_cmap('gray'))

plt.show()
    


#%% plotting a few examples:
for i in range(9,15):  
    plt.subplot(1,2,1)
    plt.title(labels[i])
    plt.imshow(plottable[i], cmap = plt.get_cmap('gray'))
    plt.subplot(1,2,2)
    plt.imshow(rec_plottable[i], cmap=plt.get_cmap('gray'))
    plt.title("reconstruction")
    plt.show()
    
    
#%%
fig, axs = plt.subplots(2,4)
z = latent[1]
add = np.array([0,0,3,0])
for i in range(desired_dim):
    axs[0,i].imshow(pc_plottable[i],cmap='gist_gray')
    axs[1,i].imshow(reconstruct(z + i * add),cmap='gist_gray')
    

plt.show()

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

#%% MSE:
MRE_PCA = np.zeros(16)
for i in range(16):
    _, _, rec = apply_pca(train_X, test_X, i+1)
    #MSE = mean_squared_error(test_X,rec)
    MRE = MeanRelativeError(test_X, rec)
    #MSE = np.mean(abs(test_X-rec))
    MRE_PCA[i] = MRE
    
MRE_ICA = np.array([23.973236957458848,
 23.810931386435218,
 23.185357914665115,
 22.6793500559618,
 22.366285176841576,
 21.859993349876632,
 21.253335767209755,
 20.8554348869842,
 20.676173241205607,
 20.450955034244796,
 20.225893801602965,
 19.97266630140722,
 19.808235951482175,
 19.65393001116622,
 19.54871925376625,
 19.450648096848056])
    
VAEMREx = np.array([1, 2, 4, 8, 16])
VAEMREy = np.array([7.821515905412953, 4.880896993302432, 4.625648590216787, 4.254391947354483 ,  3.852958951693818])
#%% 
x = np.arange(1,17,1)
plt.plot(x,MRE_ICA, color = 'red', marker='o', label = 'ICA')
plt.plot(x,MRE_PCA,color = 'blue', marker = 'o', label = 'PCA')
plt.scatter(VAEMREx,VAEMREy,color = 'green', label = 'VAE')
#plt.plot(x,re,color = 'blue', marker = 'o', label = 'PCA')
plt.legend()
plt.title("Mean Relative Error PCA")
plt.xlabel("Dimensionality of latent space")
plt.ylabel("Mean Relative Error")
plt.xlim((1,17))

#%%
MSE2 = np.zeros(13)
for i in range(13):
    _, _, rec = apply_pca(train_X, test_X, i+1)
    #MSE = mean_squared_error(test_data,rec)
    temp = np.mean(((test_X-rec)**2))
    MSE2[i] = temp
    
    
#%%
MSE3 = np.zeros(20)
_, _, rec_4 = apply_pca(train_X,test_X, 4)
for i in range(20):
    print(np.mean((test_data[i]-rec[i])**2))
    
#%%
    
for i in range(20):
    print(str(i)+", min: "+str(min(rec[i]))+", max: "+str(max(rec[i])))


#%%
N2 = (1020)
error_tot = 0.0
for i in range(N2):
    img = validation_data[i][0].to(device)
    VAE.encoder.eval()
    VAE.decoder.eval()

    with torch.no_grad():
        z_1 = VAE.encoder(img)
        rec_img_1  = VAE.decoder(z_1)

        error = (img - rec_img_1).sum()
        total_img = (img).sum()
        relative_error = error/total_img

        error_tot += (relative_error).item()

avg_error = abs(100*error_tot/(N2))
print("The avarage MSE = {}".format(avg_error))