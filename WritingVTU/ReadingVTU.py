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

def MeanError(ori,rec):
    M,N = ori.shape
    a = 0
    for i in range(M):
        for j in range(N):
            a += abs(ori[i][j] - rec[i][j])
    return a/(M*N)


def MeanRelativeError(ori,rec):
    M,N = ori.shape
    a = 0
    for i in range(M):
        for j in range(N):
            if abs(ori[i][j]) > 10e-10:
                a += abs((abs(ori[i][j] - rec[i][j]))/ori[i][j])
    return (a/(M*N)) * 100

filenameRead1  = "C:/Users/aglas/OneDrive/Bureaublad/Documenten/TW Jaar 3/CSE Minor/Final Minor Project/VTUFiles/Original/0.370044_0.849844_-0.375276/para_49.vtu"

filenameRead2  = "C:/Users/aglas/OneDrive/Bureaublad/Documenten/TW Jaar 3/CSE Minor/Final Minor Project/VTUFiles/Reconstruction/0.370044_0.849844_-0.375276/para_49.vtu"



points, cells, PointData1 = ReadVTU(filenameRead1)
points, cells, PointData2 = ReadVTU(filenameRead2)

Ori = PointData1["Displacement"]
Recon = PointData2["Displacement"]



MeanError(Ori,Recon)
MeanRelativeError(Ori,Recon)


# filenameWrite1 = "RveTestPara1Remade.vtu"

# filenameWrite2 = "RveTestPara1Remade2.vtu"


# DeleteFile(filenameWrite1)
# MakeVTUFile(points,cells,PointData1, {}, filenameWrite1)


# Disp = PointData1["Displacement"]

# PointData2 = {"Displacement":Disp}


# MakeVTUFile(points,cells,PointData2, {}, filenameWrite2)





