import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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

    def __getitem__(self, index):
        displacement_x = self.displacements[index, 0]
        displacement_y = self.displacements[index, 1]
        displacement_z = self.displacements[index, 2]

        return displacement_x, displacement_y, displacement_z

#An example data loader using the custom dataset class
training_data = DisplacementDataset("../test_data/para_1.vtu")
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)

train_displacements_x, train_displacements_y, train_displacements_z = next(iter(train_dataloader))
print(f"Displacements batch shape: {train_displacements_x.size()}")
print(f"Label: {train_displacements_x}")
