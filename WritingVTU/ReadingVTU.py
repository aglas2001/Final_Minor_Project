# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 09:51:25 2021

@author: aglas
"""
import meshio
import numpy as np
import os


def DeleteFile(filename):
    try:
        os.remove(filename)
        print("File Removed")
    except:
        print("File doesn't exist")

def MakeVTUFile(points,cells,PointData, CellData,filename): 
    ## points and cells are arrays, Point Data is a dictionary

    mesh = meshio.Mesh(
        points,
        cells,
        point_data=PointData,
        cell_data = CellData,
    )
    mesh.write(
        filename,  # str, os.PathLike, or buffer/open file
    )
    print("File is made")

    return

def ReadVTU(filename):
    mesh = meshio.read(filename)
    points, cells = mesh.points, mesh.cells
    PointData = mesh.point_data
    return points, cells, PointData

def GetPointsAndCells():
    filename  = "C:/Users/aglas/Local_Documents/GitHub/Final_Minor_Project/DataSet/rve_test/para_1.vtu"
    mesh = meshio.read(filename)
    points, cells = mesh.points, mesh.cells
    return points, cells



filenameRead  = "C:/Users/aglas/Local_Documents/GitHub/Final_Minor_Project/DataSet/rve_test/para_1.vtu"


points, cells, PointData1 = ReadVTU(filenameRead)

filenameWrite1 = "RveTestPara1Remade.vtu"

filenameWrite2 = "RveTestPara1Remade2.vtu"


DeleteFile(filenameWrite1)
MakeVTUFile(points,cells,PointData1, {}, filenameWrite1)


Disp = PointData1["Displacement"]

PointData2 = {"Displacement":Disp}


MakeVTUFile(points,cells,PointData2, {}, filenameWrite2)





