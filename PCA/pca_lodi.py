# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 23:12:26 2021

@author: jobre
"""


import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Custom Dataset loader for our displacment data
class DisplacementDataset(Dataset):
    def __init__(self, data_file):

        #Parses the XML file to retrieve the data
        #Final data saved in the data matrix with cols[x, y, z]
        temp_data = list();

        tree = ET.parse(data_file)
        root = tree.getroot()

        for data_array in root.iter("DataArray"):
            if data_array.get("Name") == "Displacement":
                temp_data = data_array.text.splitlines()

        del temp_data[0]
        del temp_data[-1]
        temp_data = np.array(temp_data)

        data = np.zeros(((len(temp_data)), 3))
       
        for i in range(len(temp_data)):
            elements = temp_data[i]
            elements = elements.strip().split()
            data[i][0] = elements[0]
            data[i][1] = elements[1]
            data[i][2] = elements[2]
                
        self.displacements = data

    def __len__(self):
        return len(self.displacements)

    def __getitem__(self):
        displacement_x = self.displacements[:, 0]
        displacement_y = self.displacements[:, 1]
        displacement_z = self.displacements[:, 2]

        return displacement_x, displacement_y, displacement_z

#%% Load and prepare data:
#An example data loader using the custom dataset class
data = DisplacementDataset("../VAE/test_data/para_1.vtu")
displacements_x, displacements_y, displacements_z = data.__getitem__()

scaler = StandardScaler()
scaler.fit(displacements_x.reshape(1,-1))
displacements_x = scaler.transform(displacements_x.reshape(1,-1))

scaler = StandardScaler()
scaler.fit(displacements_y.reshape(1,-1))
displacements_y = scaler.transform(displacements_y.reshape(1,-1))

displacements = np.concatenate((displacements_x,displacements_y),axis=0)


#%% initalize pca:
pca = PCA(0.95)

pca.fit(displacements)
reduced_dim = pca.transform(displacements)

print("dimensionality after pca:",pca.n_components_)

#%% reconstruct data:
# reconstructing the feature space with latent space:
recon = pca.inverse_transform(reduced_dim)
recon = scaler.inverse_transform(recon)






