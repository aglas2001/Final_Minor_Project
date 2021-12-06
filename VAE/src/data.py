import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

#Custom Dataset loader for our displacment data
class DisplacementDataset(Dataset):
    def __init__(self, data_file):
        data = np.loadtxt(data_file)
        self.displacements = data

    def __len__(self):
        return len(self.displacements)

    def __getitem__(self, index):
        displacement_x = self.displacements[index, 0]
        displacement_y = self.displacements[index, 1]
        displacement_z = self.displacements[index, 2]

        return displacement_x, displacement_y, displacement_z

#An example data loader using the custom dataset class
#training_data = DisplacementDataset("../test_data/para_1.txt")
#train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)

#train_displacements_x, train_displacements_y, train_displacements_z = next(iter(train_dataloader))
#print(f"Displacements batch shape: {train_displacements_x.size()}")
#print(f"Label: {train_displacements_y}")
