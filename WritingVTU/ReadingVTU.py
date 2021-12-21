# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:51:25 2021

@author: aglas
"""
import meshio
import numpy as np

import WritingVTUcomplc as WriteVTU

def ReadVTU(filename):
    mesh = meshio.read(filename)
    points, cells = mesh.points, mesh.cells
    PointData = mesh.point_data
    return points, cells, PointData

filenameRead  = "C:/Users/aglas/Local_Documents/GitHub/Final_Minor_Project/DataSet/rve_test/para_1.vtu"


points, cells, PointData = ReadVTU(filenameRead)

filenameWrite = "RveTestPara1Remade.vtu"

WriteVTU.DeleteFile(filenameWrite)
WriteVTU.MakeVTUFile(points,cells,PointData, {}, filenameWrite)