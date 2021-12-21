# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 11:04:42 2021

@author: aglas
"""

import numpy as np
# assuming data.csv is a CSV file with the 1st row being the names names for
# the columns
data = np.genfromtxt("C:/Users/aglas/Local_Documents/GitHub/Final_Minor_Project/Paraview/Testcsv.csv", dtype=None, names=True, delimiter=',', autostrip=True)
for name in data.dtype.names:
   array = data[name]

   # You can directly pass a NumPy array to the pipeline.
   # Since ParaView expects all arrays to be named, you
   # need to assign it a name in the 'append' call.
   output.RowData.append(array, name)