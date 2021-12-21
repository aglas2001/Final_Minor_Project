# -*- coding: utf-8 -*-
"""
Created on Mon Dec 20 15:34:12 2021

@author: aglas
"""

import meshio
import numpy as np
import os

def DeleteFile(filename):
    try:
        os.remove(filename)
    except:
        print("File doesn't exist")




def MakeVTUFile(points,cells):
    meshio.write_points_cells(
        filename,
        points,
        cells,
        point_data={"T": [0, 1, 2, 3]}
        # Optionally provide extra data on points, cells, etc.
        # point_data=point_data,
        # cell_data=cell_data,
        # field_data=field_data
    )
    print("File is made")
    return



filename = "SimpleSet.vtu"

DeleteFile(filename)

points = np.array([[0, 0, 0],[1, 1, 0],[-1, 1, 0],[0, 3, 0]])
cells = {"triangle": np.array([[0, 1, 2]])}

MakeVTUFile(points,cells)