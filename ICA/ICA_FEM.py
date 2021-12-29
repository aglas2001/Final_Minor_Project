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
import json


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
        displacement_z = 0
    
        return displacement_x, displacement_y,displacement_z



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
    N, D = all_data.shape
    LoopN = int(split*len(folders))
    
    train = np.zeros((N,D))
    train_folders = {"Folders":[],"Lengths": [], "StartPostion": []}
    
    count = 0
    done = []
    i=0
    while i < LoopN:
        r = random.randint(0,len(start)-2)
        if r not in done:
            p = start[r]
            q = start[r+1]
            train[count:count+(q-p)] = all_data[p:q]
            f = folders[r]
            train_folders["Folders"].append(f)
            train_folders["Lengths"].append(q-p)
            done.append(r)
            count += q-p
            i+=1
    N_test = N - count
    test = np.zeros((N_test,D))
    test_folders = {"Folders":[],"Lengths": []}
    count2 = 0
    for i in range(len(folders)-1):
        if i not in done:
            p = start[i]
            q = start[i+1]
            test[count2:count2+(q-p)] = all_data[p:q]
            f = folders[i]
            test_folders["Folders"].append(f)
            test_folders["Lengths"].append(q-p)

            count2 += q-p
    p = start[-1]
    test[count2:] = all_data[p:]
    f = folders[-1]
    test_folders["Folders"].append(f)
    test_folders["Lengths"].append(N-start[-1])
    return train[0:count,:],train_folders, test, test_folders


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

def create_vtu(para,filename): 
    Disp = np.reshape(para,(1034,2))
    Disp_xyz = np.zeros((Disp.shape[0],Disp.shape[1]+1))
    Disp_xyz[:,:-1] = Disp
    point_data = {"Displacement":Disp_xyz}    
    MakeVTUFile(points,cells,point_data, {}, filename)
    
    
    
    
def CreateComponentFiles(A, desired_dim, Path = os.getcwd()):
    for i in range (desired_dim):
        filename = Path + "VTUFiles/Components/Component_"+str(i+1)+".vtu"
        DeleteFile(filename)
        create_vtu(A[:,i].reshape(1034,2),filename)


def CreateVTUOriganalRecon(bcs, test_folders, ReconICA, Path = os.getcwd()):
    start_pos = sum(test_folders["Lengths"][:bcs])
    num_para = test_folders["Lengths"][bcs]
    Folder = test_folders["Folders"][bcs]
    
    
    PathR = Path+"VTUFiles/Reconstruction/"+Folder
    PathO = Path+"VTUFiles/Original/"+Folder

    if not os.path.exists(PathR):
        os.makedirs(PathR)
    if not os.path.exists(PathO):
        os.makedirs(PathO)
        

        
    for i in range(num_para):
        DeleteFile(Path+"VTUFiles/Reconstruction/"+Folder+"/para_"+str(i+1)+".vtu")
        DeleteFile(Path+"VTUFiles/Original/"+Folder+"/para_"+str(i+1)+".vtu")
        rec_para, true_para = get_para(start_pos,i,ReconICA)
        create_vtu(rec_para,Path+"VTUFiles/Reconstruction/"+Folder+"/para_"+str(i+1)+".vtu")
        create_vtu(true_para,Path+"VTUFiles/Original/"+Folder+"/para_"+str(i+1)+".vtu")


def ChangingComponents(ComponentToChange, frames, AmountOfComponents, ica, Path):
    COMP = np.zeros((frames,AmountOfComponents))
    L = np.linspace(-2e-3,2e-3, frames)
    COMP[:,ComponentToChange] = L
    
    ReconCOMP = ica.inverse_transform(COMP)
    PathC = Path+"VTUFiles/Components/ChangeComponent_"+str(ComponentToChange+1)

    if not os.path.exists(PathC):
        os.makedirs(PathC)
        
        
    for i in range(frames):
         DeleteFile(Path+"VTUFiles/Components/ChangeComponent_"+str(ComponentToChange+1)+"/para_"+str(i+1)+".vtu")
         create_vtu(ReconCOMP[i],Path+"VTUFiles/Components/ChangeComponent_"+str(ComponentToChange+1)+"/para_"+str(i+1)+".vtu")
    return ReconCOMP



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
    # scaler = StandardScaler()
    # scaler.fit(train)
    
    # std_train = scaler.transform(train)
    # std_test = scaler.transform(test)

    
    ica = FastICA(n_components=desired_dim,random_state=0, max_iter = 1000)
    
    
    ica.fit(train)
    SourceICA = ica.transform(test)
        
    ReconICA = ica.inverse_transform(SourceICA)
    # ReconICA = scaler.inverse_transform(ReconICA)
        
    A = ica.mixing_  # Get estimated mixing matrix
    return ica, SourceICA, ReconICA, A, SourceICA


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
trainfile = "train.txt"
testfile = "test.txt"
trainfoldersfile = "trainfolders.txt"
testfoldersfile = "testfolders.txt"

#%%

# DeleteFile(trainfile)
DeleteFile(testfile)
DeleteFile(trainfoldersfile)
DeleteFile(testfoldersfile)


# f = open(trainfile,"w")
# np.savetxt(trainfile,train)

f = open(testfile,"w")
np.savetxt(testfile,test)

json.dump(train_folders, open(trainfoldersfile,'w'))

json.dump(test_folders, open(testfoldersfile,'w'))


#%%
train = np.loadtxt(trainfile)

test = np.loadtxt(testfile)

train_folders = json.load(open(trainfoldersfile))

test_folders = json.load(open(testfoldersfile))

#%%
# desired dimentionality in latent space
desired_dim = 4

ica, SourceICA, ReconICA, A, aa  = ApplyICA(desired_dim, train, test)

#%% get point and cell data from an existing FEM solution
Path = "C:/Users/aglas/OneDrive/Bureaublad/Documenten/TW Jaar 3/CSE Minor/Final Minor Project/"
#%%
points, cells= GetPointsAndCells()

#%% make new VTU files
### Make Sure in ICA Folder
print("Make Sure in ICA Folder" )

location = "VTUFiles/Components/"
CreateComponentFiles(A, desired_dim, Path)
    

#%%
ChangingComponents(3, 50, desired_dim, ica, Path)
#%% make 50 VTU files for gif
print("Make Sure in ICA Folder" )


bcs = 5

# print(test_names[bcs])
# num_para = start[951+bcs] - start[950+bcs]
# print(num_para)
# start_pos = start[950+bcs]-split_point
# print(start_pos)

# for i in range(50):
#     DeleteFile("VTUFiles/Reconstruction/para_"+str(i+1)+".vtu", Path)
#     DeleteFile("VTUFiles/Orignal/para_"+str(i+1)+".vtu", Path)
    # rec_para, true_para = get_para(start_pos,i,ReconICA)
    # create_vtu(rec_para,"VTUFiles/Reconstruction/para_"+str(i+1)+".vtu")
    # create_vtu(true_para,"VTUFiles/Original/para_"+str(i+1)+".vtu")



CreateVTUOriganalRecon(bcs, test_folders, ReconICA,Path)


   




