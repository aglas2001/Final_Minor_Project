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
import meshio
import os
import random
import json


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
    MakeVTUFile(points,cells,point_data, {}, file_name)

#%% load data and split into training and validationd data
all_data, folder_names, start = load_all_data("../DataSet/Data_nonlinear_new/")

#split data in train and test sample
split_point = start[950]
test_names = folder_names[950:]
train, test = np.split(all_data,[split_point])

#%% arnouds way to load data

directory = "C:/Users/jobre/OneDrive - Erasmus University Rotterdam/data_arnoud/"
train = np.loadtxt(directory+"train.txt")

test = np.loadtxt(directory+"test.txt")

train_folders = json.load(open(directory+"trainfolders.txt"))

test_folders = json.load(open(directory+"testfolders.txt"))
test_names = test_folders['Folders']

#%%
# desired dimentionality in latent space
desired_dim = 4

pc, latent_space, reconstruction = apply_pca(train, test, desired_dim)

#%% get point and cell data from an existing FEM solution
filenameRead  = "../DataSet/rveLinearMultiple/para_1.vtu"
points, cells, _ = ReadVTU(filenameRead)

#%% plot principal components
# bcs = 0

# rec_para, true_para = get_para(bcs,0)
# create_vtu(rec_para,"vtu_files/reconstructed_para.vtu")
# create_vtu(true_para, "vtu_files/true_para.vtu")

for i in range (desired_dim):
    create_vtu(pc[i],"vtu_files/principal_components_pca_4/pc_"+str(i+1)+".vtu")
    
#%% make 50 VTU files for gif

bcs = 5

start_pos = 0
for i in range(bcs):
    start_pos += len(listdir('../DataSet/Data_nonlinear_new/'+test_names[i]))
    
print(test_names[bcs])
num_para = len(listdir('../DataSet/Data_nonlinear_new/'+test_names[bcs]))
print(str(num_para)+' files')

print('starting position in test/reconstruction: '+str(start_pos))

for i in range(num_para):
    rec_para, true_para = get_para(start_pos,i)
    create_vtu(rec_para,"vtu_files/reconstructed_paras/"+test_names[bcs]+"_rec_"+str(i+1)+".vtu")
    create_vtu(true_para,"vtu_files/true_paras/"+test_names[bcs]+"_true_"+str(i+1)+".vtu")
    

#%% analyse latent spaces

focus = 3
steps = 100
stepsize = 2*max(pc[focus])

# which boundary condition?
bcs = 1

start_pos = 0
for i in range(bcs):
    start_pos += len(listdir('../DataSet/Data_nonlinear_new/'+test_names[i]))
    
print(test_names[bcs])
num_para = len(listdir('../DataSet/Data_nonlinear_new/'+test_names[bcs]))
print(str(num_para)+' files')

pca = PCA(desired_dim)
scal = StandardScaler()
scal.fit(train)
pca.fit(train)
lat = pca.transform(test)

for i in range(steps):
    lat[start_pos + num_para-2] += [0,0,0,stepsize]
    rec = pca.inverse_transform(lat[start_pos + num_para-2])
    create_vtu(rec,"vtu_files/latent_space_interpretation/focus_pc["+str(focus)+"]_"+str(i)+".vtu")

#%% plot mse
mse = np.zeros(13)
for i in range(13):
    _, _, rec = apply_pca(train, test, i)
    dif = abs(rec-test)
    #mse[i] = mean_squared_error(test,rec)
    mse[i] = np.mean(dif)
    
plt.plot(mse)
plt.xlabel("Dimensionality of latent space")
plt.ylabel("Reconstruction error (Mean error)")
plt.xlim([1,12])


