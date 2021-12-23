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
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import copy as cp
import meshio
import os
import random


####
####  Dataloading
####
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
    random.shuffle(folder_names)
    total_file_count = sum([len(files) for r, d, files in walk(address)])
    num_disp = 1034*2
    
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



def RandomTrainTestSplit(split, all_data, start, folders):
    startcopy = cp.copy(start)
    folderscopy = cp.copy(folders)
    N, D = all_data.shape
    LoopN = int(split*len(folders))
    
    train = np.zeros((N,D))
    train_folders = []
    
    count = 0
    done = []
    i=0
    while i < LoopN-1:
        r = random.randint(0,len(startcopy)-2)
        if r not in done:
            p = startcopy[r]
            q = startcopy[r+1]
            train[count:count+(q-p)] = all_data[p:q]
            f = folderscopy[r]
            train_folders.append(f)
            done.append(r)
            count += q-p
            i+=1
    N_test = N - count
    test = np.zeros((N_test,D))
    test_folders = []
    count2 = 0
    for i in range(len(folders)-1):
        if i not in done:
            p = startcopy[i]
            q = startcopy[i+1]
            test[count2:count2+(q-p)] = all_data[p:q]
            f = folderscopy[i]
            test_folders.append(f)
            count2 += q-p
    p = startcopy[-1]
    test[count2:] = all_data[p:]
    f = folderscopy[-1]
    train_folders.append(f)
    return train[0:count,:],train_folders, test,test_folders








### Getting Parameters

def get_para(start,para,reconstruction):
    rec_para = reconstruction[start+para]
    true_para = test[start+para]
    return rec_para, true_para

#start is first position in reconstruction and test!
def get_all_paras(start,num_paras,reconstruction):
    
    rec_paras = np.zeros((num_paras,2*1034))
    true_paras = np.zeros((num_paras,2*1034))
    for i in range(num_paras):
        rec_paras[i] = reconstruction[start+i]
        true_paras[i] = test[start+i]
        
    rec = np.zeros((rec_paras.shape[0],rec_paras.shape[1]+1))
    rec[:,:-1] = rec_paras
    
    true = np.zeros((true_paras.shape[0],true_paras.shape[1]+1))
    true[:,:-1] = true_paras

    return rec, true
    

##
#### VTU-files
##
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
    filename  = "../DataSet/rve_test/para_1.vtu"
    mesh = meshio.read(filename)
    points, cells = mesh.points, mesh.cells
    return points, cells

def create_vtu(para,file_name): 
    Disp = np.reshape(para,(1034,2))
    Disp_xyz = np.zeros((Disp.shape[0],Disp.shape[1]+1))
    Disp_xyz[:,:-1] = Disp
    point_data = {"Displacement":Disp_xyz}    
    MakeVTUFile(points,cells,point_data, {}, file_name)
    
    
    
    
def CreateComponentFiles(A, desired_dim, location):
    for i in range (desired_dim):
        filename = location + "Component_"+str(i+1)+".vtu"
        DeleteFile(filename)
        create_vtu(A[:,i].reshape(1034,2),filename)

##    
#### ICA
##
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


def ApplyICA(desired_dim, train, test):
    scaler = StandardScaler()
    scaler.fit(train)
    
    ica = FastICA(n_components=desired_dim,random_state=0, max_iter = 1000)
    
    
    ica.fit(train)
    SourceICA = ica.transform(test)

        
    ReconICA = ica.inverse_transform(SourceICA)
    ReconICA = scaler.inverse_transform(ReconICA)
        
    A = ica.mixing_  # Get estimated mixing matrix
    return ica,SourceICA,ReconICA, A


#%% load data and apply pca
print("Make Sure in ICA Folder" )

all_data, folder_names, start = load_all_data("../DataSet/Data_nonlinear/")

#%%
Helpall_data = cp.copy(all_data)
Helpfoldernames = cp.copy(folder_names)
helpstart = cp.copy(start)

#%%
all_data = cp.copy( Helpall_data)
folder_names = cp.copy( Helpfoldernames)
start = cp.copy( helpstart)
#%%
#split data in train and test sample




# split_point = start[950]
# test_names = folder_names[950:]
# train, test = np.split(all_data,[split_point])

train,train_folders, test,test_folders = RandomTrainTestSplit(0.8, all_data, start,folder_names)



#%%
# desired dimentionality in latent space
desired_dim = 4

ica, SourceICA, ReconICA, A = ApplyICA(desired_dim, train, test)

#%% get point and cell data from an existing FEM solution
filenameRead  = "../DataSet/rve_test/para_1.vtu"
points, cells, _ = ReadVTU(filenameRead)

#%% make new VTU files
### Make Sure in ICA Folder
print("Make Sure in ICA Folder" )

location = "VTUFiles/Components/"
CreateComponentFiles(A, desired_dim, location)
    
#%% make 50 VTU files for gif
print("Make Sure in ICA Folder" )

bcs = 0

print(test_names[bcs])
num_para = start[951+bcs] - start[950+bcs]
print(num_para)
start_pos = start[950+bcs]-split_point
print(start_pos)

for i in range(num_para):
    DeleteFile("VTUFiles/Reconstruction/para_"+str(i+1)+".vtu")
    DeleteFile("VTUFiles/Original/para_"+str(i+1)+".vtu")
    rec_para, true_para = get_para(start_pos,i,ReconICA)
    create_vtu(rec_para,"VTUFiles/Reconstruction/para_"+str(i+1)+".vtu")
    create_vtu(true_para,"VTUFiles/Original/para_"+str(i+1)+".vtu")
    




