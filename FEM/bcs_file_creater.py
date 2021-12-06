#create files
#[xstretch, ystretch, xshear from y]
import os
import shutil
import subprocess, sys
import numpy.random as npr
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import xml.etree.ElementTree as ET
npr.seed(113)

os.chdir('.\dockerjive\dockerjive')
cwd = os.getcwd()
print(os.getcwd())

def create_file(bcs): #create new folder with bcs
    source = r'C:\Users\adria\Desktop\CSE_Minor_Project\dockerjive\dockerjive\rve'
    destination_folder = r'C:\Users\adria\Desktop\CSE_Minor_Project\dockerjive\Newfolder\\'
    #bcs = bcs[1:]
    destination = destination_folder + bcs
    shutil.copytree(source, destination)

def change_bcs(bcs, bcs2): #change bcs file to target values
    #reach target folder
    #print("Creating for ", bcs)
    os.chdir('..')
    destination_folder = r".\Newfolder\\"
    bcs = bcs[1:]
    destination = destination_folder + bcs2
    os.chdir(destination)
    #change to txt file
    
    thisFile = "./bcs"
    base = os.path.splitext(thisFile)[0]
    os.rename(thisFile, base + ".txt")
    
    with open('bcs.txt', 'r') as file:
        data = file.readlines()
    data[2] = 'unitVec = [' + bcs + ';\n'
    with open('bcs.txt', 'w') as file:
        file.writelines( data )
    os.rename(base + ".txt", thisFile)
    os.chdir("..")

def rename_file(bcs, bcs2):
    files_location = r'C:\Users\adria\Desktop\CSE_Minor_Project\dockerjive\Newfolder'
    os.chdir(files_location)
    os.rename(bcs2, bcs)

    
def rename(target):
    target1 = target[0] + '_' + target[1] + '_' + target[2]
    target2 = ",".join(target)
    target2 = target2 = "[" + target2 + "]"
    return target2, target1

def main():
    status = 600
    for i in range(0,status*3):
        rnd = npr.normal()
    size = 400
    for i in range(0,size):
        xstretch = npr.normal()
        ystretch = npr.normal()
        shear = npr.normal()
        size = math.sqrt(xstretch**2+ystretch**2+shear**2)
        xstretch = round(xstretch/size, 6)
        ystretch = round(ystretch/size, 6)
        shear = round(shear/size,6)
        target = [str(xstretch), str(ystretch), str(shear)]
        target1, target2 = rename(target)
        create_file(target2)
        change_bcs(target1, target2)
        
def visualise():
    vector_correct = []
    vector_wrong = []
    for folder in os.listdir(r'C:\Users\adria\Desktop\CSE_Minor_Project\dockerjive\Newfolder\\'):
        bcs = folder.split("_")
        directory = r'C:\Users\adria\Desktop\CSE_Minor_Project\dockerjive\Newfolder\\' + folder
        num_files = len(os.listdir(directory))
        if num_files == 50:
            vector_correct.append([0,0,0,float(bcs[0]), float(bcs[1]), float(bcs[2])])
            #print("correct", float(bcs[0]) *float(bcs[1])*float(bcs[2]))
        else:
            vector_wrong.append([0,0,0,float(bcs[0]), float(bcs[1]), float(bcs[2])])
            #print("wrong", float(bcs[0]) *float(bcs[1])*float(bcs[2]))
    vector_correct = np.array(vector_correct)
    vector_wrong = np.array(vector_wrong)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z, U, V, W = zip( * vector_correct)
    ax.quiver(X,Y,Z,U,V,W, color = 'b')
    X, Y, Z, U, V, W = zip( * vector_wrong)
    ax.quiver(X,Y,Z,U,V,W, color = 'r')
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('x stretch')
    ax.set_ylabel('y stretch')
    ax.set_zlabel('xy shear')
    ax.view_init(elev=0, azim=90)
    plt.show()


def clear():
    for folder in os.listdir(r'C:\Users\adria\Desktop\CSE_Minor_Project\dockerjive\Newfolder\\'):
        bcs = folder.split("_")
        target = [bcs[0], bcs[1], bcs[2]]
        target1, target2 = rename(target)
        rename_file(target1,target2)
    

def file_strip():
    for folder in os.listdir(r'C:\Users\adria\Desktop\CSE_Minor_Project\dockerjive\Newfolder\\'):
        target = r'C:\Users\adria\Desktop\CSE_Minor_Project\dockerjive\Newfolder\\' + folder
        os.chdir(target)
        for file in glob.glob("*.vtu"):
            tree = ET.parse(file)
            root = tree.getroot()
            for data_array in root.iter("DataArray"):
                if data_array.get("Name") == "Displacement":
                   temp_data = data_array.text.splitlines()

            del temp_data[0]
            del temp_data[-1]
            temp_data = np.array(temp_data)

            data = np.zeros(((len(temp_data)), 3))
       
            for i in range(len(temp_data)):
                elements = temp_data[i]
                elements = elements.strip().split()
                data[i][0] = elements[0]
                data[i][1] = elements[1]
                data[i][2] = elements[2]
    
            file = file[:-3] + 'txt'
            with open(file, 'w') as f:
                for i in range(len(temp_data)):
                    f.write(str(data[i][0]) +' '+  str(data[i][1]) +' '+ str(data[i][2]) + '\n')
                    
        for file in (set(glob.glob("*")) - set(glob.glob("*.txt"))):
            if os.path.isdir(file):
                target = target +"\\"+  file
                shutil.rmtree(target, ignore_errors=True)
            else:
                os.remove(file)
        

#main() #Creates files with new naming scheme for powershell script
visualise() #Visualise Dimensions covered
#file_strip() #Strip unnecessary data
#clear() #Renames files to array (Prob not neccesary causes too many complications)

