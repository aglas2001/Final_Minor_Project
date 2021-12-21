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

    mesh = meshio.Mesh(
        points,
        cells,
        # Optionally provide extra data on points, cells, etc.
        point_data=PointData,
        # Each item in cell data must match the cells array
        cell_data = CellData,
    )
    mesh.write(
        filename,  # str, os.PathLike, or buffer/open file
        # file_format="vtk",  # optional if first argument is a path; inferred from extension
    )
    print("File is made")

    return






filename = "ComplcSet.vtu"
    
k = 6    
    
    
Triangles = np.zeros((k,3))
for i in range(k-2):
    Triangles[i] = [i,i+1,i+2]
    

points = np.array([
    [1, 1],
    [1, 4],
    [3, 2],
    [2, 5],
    [5, 7],
    [9, 2],
    [0, 0],
    [0,10],
    [10,0],
    [10,10]
])
cells = [
    ("triangle", Triangles),
    ("quad", [[6,7,9,8]]),
]

a = np.zeros(k)
for i in range(k):
    a[i] = i
m = [np.mean(a)]

PointData={"T": [0.3, -1.2, 0.5, 0.7, 0.0, -3.0,  0,0,0,0]}
CellData={"a": [a, m]}


DeleteFile(filename)
MakeVTUFile(points,cells,PointData,CellData,filename)



