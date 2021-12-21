# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 19:14:19 2021

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

def apply_pca(train, test, desired_dim):
    
    scaler = StandardScaler()
    scaler.fit(train)
    
    #standardize data
    std_data = scaler.transform(train)
    std_test = scaler.transform(test)
    
    pca = PCA(n_components=desired_dim)
    pca.fit(std_data)
    pc = pca.components_
    latent = pca.transform(std_test)
    rec_data = scaler.inverse_transform(pca.inverse_transform(latent))
    
    return pc, latent, rec_data


def get_para(bcs,para):
    rec_para = reconstruction[bcs*50:((bcs+1)*50)][para]
    true_para = test[bcs*50:((bcs+1)*50)][para]
    return rec_para, true_para
    

#%%
all_data = load_all_data("../DataSet/Data_nonlinear/")

train, test = np.split(all_data,[40000])
desired_dim = 4

pc, latent_space, reconstruction = apply_pca(train, test, desired_dim)







folder_names = next(walk("../DataSet/Data_nonlinear/"), (None, None, []))[1]
