#This only works if the data.py file is in the same folder as your file
from data import DisplacementDataset
import numpy as np

dataset = DisplacementDataset("../../Dataset/Data_nonlinear/", ratio=0.2, seed=0, train=True, tensor=False)

displacement_matrix = np.zeros((len(dataset), len(dataset[0])))

count = 0
for timestep in dataset:
	displacement_matrix[count] = timestep
	count += 1