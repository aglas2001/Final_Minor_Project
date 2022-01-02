# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 23:12:38 2021

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
import meshio
import os
import random


#Custom Dataset loader for our displacment data
class DisplacementDataset(Dataset):
    def __init__(self, data_folder):
        file_count = len(listdir(data_folder))

        displacement_matrix = np.zeros((file_count, 1636*2))

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
    random.shuffle(folder_names)
    total_file_count = sum([len(files) for r, d, files in walk(address)])
    num_disp = 1636*2
    
    all_data = np.zeros((total_file_count,num_disp))
    start = []
    count = 0
    
    for i in range(len(folder_names)):
        start.append(int(count))
        temp = DisplacementDataset(address+folder_names[i]).displacements
        num_files = temp.shape[0]
        
        for j in range(num_files):
            all_data[count] = temp[j,:]   
            count += 1
            
    return all_data, folder_names, start

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


def get_para(start,para):
    rec_para = reconstruction[start+para]
    true_para = test[start+para]
    return rec_para, true_para

#start is first position in reconstruction and test!
def get_all_paras(start,num_paras):
    
    rec_paras = np.zeros((num_paras,2*1636))
    true_paras = np.zeros((num_paras,2*1636))
    for i in range(num_paras):
        rec_paras[i] = reconstruction[start+i]
        true_paras[i] = test[start+i]
        
    rec = np.zeros((rec_paras.shape[0],rec_paras.shape[1]+1))
    rec[:,:-1] = rec_paras
    
    true = np.zeros((true_paras.shape[0],true_paras.shape[1]+1))
    true[:,:-1] = true_paras

    return rec, true
    
def DeleteFile(filename):
    try:
        os.remove(filename)
        print("File Removed")
    except:
        print("File doesn't exist")

def MakeVTUFile(points,cells,PointData, CellData,filename): 
    ## points and cells are arrays, Point Data is a dictionary

    mesh = meshio.Mesh(
        points,
        cells,
        point_data=PointData,
        cell_data = CellData,
    )
    mesh.write(
        filename,  # str, os.PathLike, or buffer/open file
    )
    print("File is made")

    return

def ReadVTU(filename):
    mesh = meshio.read(filename)
    points, cells = mesh.points, mesh.cells
    PointData = mesh.point_data
    return points, cells, PointData

def GetPointsAndCells():
    filename  = "C:/Users/jobre/OneDrive - Erasmus University Rotterdam/Github_cloned/Final_Minor_Project/DataSet/rve_test/para_1.vtu"
    mesh = meshio.read(filename)
    points, cells = mesh.points, mesh.cells
    return points, cells

def create_vtu(para,file_name): 
    Disp = np.reshape(para,(1636,2))
    Disp_xyz = np.zeros((Disp.shape[0],Disp.shape[1]+1))
    Disp_xyz[:,:-1] = Disp
    point_data = {"Displacement":Disp_xyz}    
    #print(point_data)
    MakeVTUFile(points,cells,point_data, {}, file_name)
    
    
#%% load data and apply pca
all_data, folder_names, start = load_all_data("../DataSet/Data_linear/")

#split data in train and test sample
split_point = start[950]
test_names = folder_names[950:]
train, test = np.split(all_data,[split_point])


#%% get point and cell data from an existing FEM solution
filenameRead  = "../DataSet/rveLinearMultiple/para_1.vtu"
points, cells, _ = ReadVTU(filenameRead)

#%%
# desired dimentionality in latent space
desired_dim = 3

pc, latent_space, reconstruction = apply_pca(train, test, desired_dim)

#%%
bcs = 5

print(test_names[bcs])
num_para = start[951+bcs] - start[950+bcs]
print(num_para)
start_pos = start[950+bcs]-split_point
print(start_pos)

for i in range(num_para):
    rec_para, true_para = get_para(start_pos,i)
    create_vtu(rec_para,"vtu_files/reconstructed_paras_lin/rec_"+str(test_names[bcs])+".vtu")
    create_vtu(true_para,"vtu_files/true_paras_lin/true_"+str(test_names[bcs])+".vtu")
    
#%% plot principal components

for i in range(desired_dim):
    create_vtu(pc[i],"vtu_files/lin_pc_"+str(i+1)+".vtu")
    
    
#%% mse for different dimensionalities


mse = np.zeros(13)
for i in range(13):
    pca = PCA(n_components=i+1)
    scal = StandardScaler()
    scal.fit(train)
    pca.fit(train)
    lat = pca.transform(test)
    rec = scal.inverse_transform(pca.inverse_transform(lat))
    mse[i] = mean_squared_error(test,rec)
    
plt.plot(mse)
plt.xlabel("Dimensionality of latent space")
plt.ylabel("Reconstruction error (MSE)")
plt.xlim([1,12])
