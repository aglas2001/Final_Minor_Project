import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, isdir, join


#Custom Dataset loader for our displacment data
class DisplacementDataset(Dataset):
    def __init__(self, data_folder, ratio, size, seed, train, tensor):
        self.tensor = tensor

        np.random.seed(seed)

        folder_list = list()
        for folder in (listdir(data_folder)):
            if isdir(join(data_folder, folder)):
                folder_list.append(join(data_folder, folder))

        folder_list = np.random.choice(np.array(folder_list), int(size*len(folder_list)), replace=False)

        train_folders = np.random.choice(folder_list, int((1-ratio)*len(folder_list)), replace=False)
        validation_folders = np.setxor1d(folder_list, train_folders)

        

        if train:
            train_data = list()

            for folder in train_folders:
                for data_file in listdir(join(data_folder, folder)):
                    if isfile(join(join(data_folder, folder), data_file)):
                        train_data.append(join(join(data_folder, folder), data_file))
            self.displacements = np.array(train_data)
        else:
            validation_data = list()

            for folder in validation_folders:
                for data_file in listdir(join(data_folder, folder)):
                    if isfile(join(join(data_folder, folder), data_file)):
                        validation_data.append(join(join(data_folder, folder), data_file))
            self.displacements = np.array(validation_data)

        #print(len(self.displacements))
    def __len__(self):
        return len(self.displacements)

    def __getitem__(self, index):
        displacement = np.loadtxt(self.displacements[index], usecols=(0,1))
        displacement = displacement.flatten()
        if self.tensor:
            displacement = torch.from_numpy(displacement)
            displacement = displacement.type(torch.FloatTensor)

        return displacement

#An example data loader using the custom dataset class
#training_data = DisplacementDataset("../test_data/para_1.txt")
#train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)

#train_displacements_x, train_displacements_y, train_displacements_z = next(iter(train_dataloader))
#print(f"Displacements batch shape: {train_displacements_x.size()}")
#print(f"Label: {train_displacements_y}")

#DisplacementDataset("../../Dataset/Data_nonlinear_new/", ratio=0.2, seed=0, train=False)