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
    start = np.zeros(len(folder_names))
    count = 0
    
    for i in range(len(folder_names)):
        start[i] = count
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
    pc = pca.components_
    latent = pca.transform(std_test)
    rec_data = scaler.inverse_transform(pca.inverse_transform(latent))
    
    return pc, latent, rec_data


def get_para(bcs):
    rec_para = reconstruction[bcs]
    true_para = test[bcs]
    return rec_para, true_para

    
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
    point_data = {"Displacement":Disp}
    MakeVTUFile(points,cells,point_data, {}, file_name)

#%% load data and apply pca
all_data, folder_names, start = load_all_data("../DataSet/Data_linear/")

#split data in train and test sample
train, test = np.split(all_data,[900])

# desired dimentionality in latent space
desired_dim = 3

pc, latent_space, reconstruction = apply_pca(train, test, desired_dim)

#%% get point and cell data from an existing FEM solution
filenameRead  = "../DataSet/rveLinearMultiple/para_1.vtu"
points, cells, _ = ReadVTU(filenameRead)
    
#%% make new vtu files

bcs = 28

print(folder_names[900+bcs])

rec_para, true_para = get_para(bcs)
create_vtu(rec_para,"vtu_files/reconstructed_paras_lin/rec_"+str(folder_names[900+bcs])+".vtu")
create_vtu(true_para,"vtu_files/true_paras_lin/true_"+str(folder_names[900+bcs])+".vtu")

#%% plot principal components

for i in range(desired_dim):
    create_vtu(pc[i],"vtu_files/lin_pc_"+str(i+1)+".vtu")

