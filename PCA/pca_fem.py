# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 12:59:25 2021

@author: jobre
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os import listdir, walk
from os.path import isfile, join
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt


#Custom Dataset loader for our displacment data
class DisplacementDataset(Dataset):
    def __init__(self, data_folder):
        file_count = len(listdir(data_folder))

        displacement_matrix = np.zeros((file_count, 1034*2))

        def order(input):
            return int(input.strip("para_.txt"))

        count = 0

        for data_file in sorted(listdir(data_folder), key=order):
            if isfile(join(data_folder, data_file)):
                data = np.loadtxt(join(data_folder, data_file), usecols=(0,1))
                data = data.flatten()
                displacement_matrix[count] = data
                count += 1
        self.displacements = displacement_matrix
        #print(self.displacements)

    def __len__(self):
        return len(self.displacements)

    def __getitem__(self, index):
        displacement_x = self.displacements[index, 0]
        displacement_y = self.displacements[index, 1]
    
        return displacement_x, displacement_y

#An example data loader using the custom dataset class
#training_data = DisplacementDataset("../test_data/para_1.txt")
#train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)

#train_displacements_x, train_displacements_y, train_displacements_z = next(iter(train_dataloader))
#print(f"Displacements batch shape: {train_displacements_x.size()}")
#print(f"Label: {train_displacements_y}")

#%% load data, takes a while

def load_all_data(address):

    folder_names = next(walk(address), (None, None, []))[1]
    total_file_count = sum([len(files) for r, d, files in walk(address)])
    num_disp = 1034*2
    
    all_data = np.zeros((total_file_count,num_disp))
    count = 0
    
    for i in range(len(folder_names)):
        temp = DisplacementDataset(address+folder_names[i]).displacements
        num_files = [len(files) for r, d, files in walk(address+folder_names[i])][0]
        
        for j in range(num_files):
            all_data[count] = temp[j,:]   
            count += 1
            
    return all_data

all_data = load_all_data("../DataSet/Data_nonlinear/")


# print("../DataSet/Data_nonlinear/"+folder_names[1])

#%% standardize data and set parameters
scaler = StandardScaler()

scaler.fit(all_data)

train, test = np.split(all_data,[40000])

#standardize data
std_data = scaler.transform(train)
std_test = scaler.transform(test)

N, D = std_data.shape
desired_dim = 4

#%% apply pca
pca = PCA(n_components=desired_dim)
pca.fit(std_data)
latent = pca.transform(std_test)

#%% reconstuction

rec_data = scaler.inverse_transform(pca.inverse_transform(latent))

mean_squared_error(test,rec_data)

# plt.imshow(rec_data[0,:])


#%% plot reconstruction error for multiple latent dimensionalities

mse = np.zeros(13)
for i in range(13):
    pca = PCA(n_components=i+1)
    pca.fit(std_data)
    lat = pca.transform(std_test)
    rec = scaler.inverse_transform(pca.inverse_transform(lat))
    mse[i] = mean_squared_error(test,rec)


plt.plot(mse)
plt.xlabel("Dimensionality of latent space")
plt.ylabel("Reconstruction error (MSE)")
plt.xlim([1,12])


#%%

one_para = rec_data[0:50][0]

plt.scatter(one_para[:1034], one_para[1034:])



real_one_para = test[0:50][0]

plt.scatter(real_one_para[:1034], real_one_para[1034:])

#%%
dif = one_para - real_one_para

