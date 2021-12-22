import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import isfile, isdir, join


#Custom Dataset loader for our displacment data
class DisplacementDataset(Dataset):
    def __init__(self, data_folder, ratio, seed, train):

        np.random.seed(seed)

        file_list = list()
        for folder in (listdir(data_folder)):
            if isdir(join(data_folder, folder)):
                for data_file in listdir(join(data_folder, folder)):
                    if isfile(join(join(data_folder, folder), data_file)):
                        file_list.append(join(join(data_folder, folder)))

        #train_folders = np.zeros((1-ratio)*len(folder_list))
        #validation_folders = np.zeros(ratio*len(folder_list))

        train_folders = np.random.choice(np.array(file_list), int((1-ratio)*len(file_list)), replace=False)
        validation_folders = np.setxor1d(np.array(file_list), train_folders)

        if train:
            self.displacements = train_folders
        else:
            self.displacements = validation_folders


        #def order(input):
        #    return int(input.strip("para_.txt"))

        #array_length = 0

        # for folder in (listdir(data_folder)):
        #     if isdir(join(data_folder, folder)):
        #         for data_file in listdir(join(data_folder, folder)):
        #             if isfile(join(join(data_folder, folder), data_file)):
        #                 array_length += 1

        # displacement_matrix = np.zeros((array_length, 1034*2))

        # count = 0

        #for folder in (listdir(data_folder)):
        #    if isdir(join(data_folder, folder)):
        #        file_count = len(listdir(data_folder))

        #        sub_displacement_matrix = np.zeros((file_count, 1034*2))


        #        for data_file in sorted(listdir(join(data_folder, folder)), key=order):
        #            if isfile(join(join(data_folder, folder), data_file)):
        #                data = np.loadtxt(join(join(data_folder, folder), data_file), usecols=(0,1))
        #                data = data.flatten()
        #                displacement_matrix[count] = data
        #                count += 1

        #self.displacements = np.array(displacement_matrix)
    def __len__(self):
        return len(self.displacements)

    def __getitem__(self, index):
        displacement = np.loadtxt(self.displacements[index], usecols=(0,1))
        displacement = displacement.flatten()

        return displacement

#An example data loader using the custom dataset class
#training_data = DisplacementDataset("../test_data/para_1.txt")
#train_dataloader = DataLoader(training_data, batch_size=64, shuffle=False)

#train_displacements_x, train_displacements_y, train_displacements_z = next(iter(train_dataloader))
#print(f"Displacements batch shape: {train_displacements_x.size()}")
#print(f"Label: {train_displacements_y}")

DisplacementDataset("../../Dataset/Data_nonlinear/", ratio=0.2, seed=0, train=True)