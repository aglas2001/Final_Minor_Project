# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:47:55 2021

@author: jobre
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 11:24:29 2021

@author: jobre
"""

import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from matplotlib import pyplot as plt

#Custom Dataset loader for our displacement data
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
displacements_x = np.zeros((50,1034))
displacements_y = np.zeros((50,1034))
displacements_z = np.zeros((50,1034))
for i in range(0,50):
    data = DisplacementDataset("../DataSet/[0.0,1.0,1.0]/para_"+str(i+1)+".vtu")
    displacements_x[i,:], displacements_y[i,:], displacements_z[i,:] = data.__getitem__()
    


#%% Plot load-displacement curve:

test = displacements_x[:,8]
    
plt.plot(test)
    
