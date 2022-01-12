# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 11:22:09 2021

@author: aglas
"""

from sklearn.datasets import fetch_openml
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import math as m
import copy
import matplotlib.animation as animation
import os




def loadmnist():
    mnist_dataset = fetch_openml('mnist_784')
    return mnist_dataset


def loaddata(mnist_dataset):
    mnist = mnist_dataset.data.to_numpy()
          
    scaler = StandardScaler()
    scaler.fit(mnist)
    data = scaler.transform(mnist)
    N, D = data.shape
    mu = np.mean(data,axis=0)
    
    plottable = np.reshape(mnist,(N,28,28))
    labels = mnist_dataset.target.array
    return data, scaler, N, D, mu,labels,plottable

def loaddatasplit(mnist_dataset, split):
    # mnist = mnist_dataset.data.to_numpy()
    N, D = mnist_dataset.data.shape
    # test_size: what proportion of original data is used for test set
    train_img, test_img, train_lbl, test_lbl = train_test_split(
        mnist_dataset.data, mnist_dataset.target, test_size=split, random_state=0)
    
    #preserve orginial values train_img for plot later:
    OrignalPlot = np.reshape(test_img.to_numpy(),(int(N*split),28,28))
    Labels = test_lbl.array

    # Fit on training set only. Computes the mean and std to be used for later scaling.
    scaler = StandardScaler()
    scaler.fit(train_img)

    # Applies standardization to both the training set and the test set.
    return train_img,test_img,OrignalPlot,Labels,scaler, N, D



def loaddataOneNumber(mnist_dataset, split, Number):
    DataArray = mnist_dataset.data.to_numpy()
    N, D = mnist_dataset.data.shape
    NewN = 0
    lst = []
    for i in range(N):
        if mnist_dataset.target[i] == str(Number):
            lst.append(i)
            NewN += 1
    OnlyNumber = np.zeros((NewN,D))
    counter = 0
    for i in lst:
        OnlyNumber[counter] = DataArray[i]
        counter += 1
    int(NewN*split)
    
    SplitPoint = int(NewN*(1-split))
    
    train_img = OnlyNumber[0:SplitPoint]
    test_img = OnlyNumber[SplitPoint:NewN]

    #preserve orginial values train_img for plot later:
    OrignalPlot = np.reshape(test_img,(NewN - SplitPoint,28,28))
    # Fit on training set only. Computes the mean and std to be used for later scaling.
    scaler = StandardScaler()
    scaler.fit(train_img)
    
    data = scaler.transform(train_img)
    test_data = scaler.transform(test_img)
    
    # Applies standardization to both the training set and the test set.
    return data,test_data,OrignalPlot,scaler, NewN, D, SplitPoint







def ApplyICA(desired_dim, scaler, train_img, test_img, N):
    ica = FastICA(n_components=desired_dim,random_state=0, max_iter = 1000)
    
    
    ica.fit(train_img)
    SourceICA = ica.transform(test_img)

        
    ReconICA = ica.inverse_transform(SourceICA)
    ReconICA = scaler.inverse_transform(ReconICA)
    ReconICAPlottable = np.reshape(ReconICA,(N,28,28))
        
    A = ica.mixing_  # Get estimated mixing matrix
    return ica,SourceICA, ReconICAPlottable, A
    

def ReconstructionError(plottable,recpca_plottable,recica_plottable,labels):
    rmseICA = m.sqrt(mean_squared_error(plottable, recica_plottable))
    nrmseICA = rmseICA/m.sqrt(np.mean(plottable**2))
    return rmseICA,nrmseICA

def PlotReconstructionOneNumber(NumbertoPlot, plottable,ReconICAPlottable,labels):
    numberstr = str(NumbertoPlot)
    i = 0
    while labels[i] != numberstr:
        i+=1
    titlestr = "Reconstructing " + str(labels[i])+ " using ICA"
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(titlestr, fontsize=16)
    axs[0].imshow(plottable[i], cmap = plt.get_cmap('gray'))
    axs[0].set_title("Original")
                
    axs[1].imshow(ReconICAPlottable[i], cmap=plt.get_cmap('gray'))
    axs[1].set_title("Reconstruction")
    


  
    
def PlotReconstructionMultiple(AmountOfPlots, plottable,ReconICAPlottable,labels):
    for i in range(AmountOfPlots):
        titlestr = "Reconstructing " + str(labels[i])+ " using ICA"
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(titlestr, fontsize=16)
        axs[0].imshow(plottable[i], cmap = plt.get_cmap('gray'))
        axs[0].set_title(labels[i])
        axs[1].imshow(ReconICAPlottable[i], cmap=plt.get_cmap('gray'))
        axs[1].set_title("ICA recon")    

def PlotReconstructionICAOneN(AmountOfPlots, plottable,ReconICAPlottable,Number):
    for i in range(AmountOfPlots):
        titlestr = "Reconstructing " + str(Number)+  " using ICA"
        fig, axs = plt.subplots(1, 2)
        fig.suptitle(titlestr, fontsize=16)
        axs[0].imshow(plottable[i], cmap = plt.get_cmap('gray'))
        axs[0].set_title("Original")
        axs[1].imshow(ReconICAPlottable[i], cmap=plt.get_cmap('gray'))
        axs[1].set_title("Reconstruction")

def PlotComponents(AmountOfComponents, A):
    fig, axs = plt.subplots(1, AmountOfComponents)
    fig.suptitle("Components", fontsize=16)
    for j in range(AmountOfComponents):
        Comp = np.reshape(A[:,j],(28,28))
        axs[j].imshow(Comp, cmap = plt.get_cmap('gray'))
        titlestr = "Component " + str(j)
        axs[j].set_title(titlestr)

def ChangingComponent(AmountOfComponents, ComponentToChange, N, ica, value, scaler):
    COMP = np.zeros((1,AmountOfComponents))
    COMP[0][ComponentToChange-1] = value
        
    ReconICAComp = ica.inverse_transform(COMP)
    ReconICAComp = scaler.inverse_transform(ReconICAComp)
    ReconICACompPlottable = np.reshape(ReconICAComp,(28,28))

    # fig, axs = plt.subplots(1,1)
    # axs[0].imshow(ReconICACompPlottable, cmap = plt.get_cmap('gray'))
    return ReconICACompPlottable

def ChangeComponents(AmountOfComponents, ComponentToChange,Changes, N, ica, values,scaler):
    COMP = np.zeros((Changes,AmountOfComponents))
    for i in range(Changes):
        COMP[i][ComponentToChange-1] = values[i]
        
    ReconICAComp = ica.inverse_transform(COMP)
    ReconICAComp = scaler.inverse_transform(ReconICAComp)
    ReconICACompPlottable = np.reshape(ReconICAComp,(Changes,28,28))
   
    
    titlestr = "Changing Component " + str(ComponentToChange)
    fig, axs = plt.subplots((Changes-1)//4 + 1, 4)
    fig.suptitle(titlestr, fontsize=16)
    
    for i in range(Changes):
        titlstr = "Comp = " + str(values[i])
        axs[i].imshow(ReconICACompPlottable[i], cmap = plt.get_cmap('gray'))
        axs[i].set_title(titlstr)



def ShowComponents(AmountOfComponents, A):
    for i in range(AmountOfComponents):
        Comp = np.reshape(A[:,i],(28,28))
        plt.imshow(Comp, cmap = plt.get_cmap('gray'))
        plt.show()

def MeanError(test,rec):
    M,N = test.shape
    a = 0
    for i in range(M):
        for j in range(N):
            a += abs(test[i][j] - rec[i][j])
    return a/(M*N)

def ErrorDifferentDimens(S, L,scaler, train_img, test_img, N, OrignalPlot):
    mse = []
    ME = []
    for i in range(L-S):
        print(S+i)
        ica, SourceICA, helperee , A = ApplyICA(S+i, scaler, train_img, test_img, N)
        # PlotReconstructionOneNumber(7, OrignalPlot,ReconICAPlottable,Labels)
        mse.append(mean_squared_error(OrignalPlot.reshape(N,784),helperee.reshape(N,784)))
        ME.append(MeanError(OrignalPlot.reshape(N,784),helperee.reshape(N,784)))
    
    plt.plot(mse)
    plt.xlabel("Dimensionality of latent space")
    plt.ylabel("Reconstruction error (MSE)")
    plt.xlim([S,L])
    plt.show()
    
    
    plt.plot(ME)
    plt.xlabel("Dimensionality of latent space")
    plt.ylabel("Reconstruction error (ME)")
    plt.xlim([S,L])
    plt.show()
    return mse, ME
    

def DeleteFile(filename):
    try:
        os.remove(filename)
        print("File Removed")
    except:
        print("File doesn't exist")

#%%
datasplit = 1/7
desired_dim=4

mnist_dataset = loadmnist()
#%%
train_img,test_img,OrignalPlot,Labels,scaler, N, D = loaddatasplit(mnist_dataset,datasplit)


#%%
ica,SourceICA, ReconICAPlottable, A = ApplyICA(desired_dim, scaler, train_img, test_img, int(datasplit*N))
#%%
PlotReconstructionOneNumber(3, OrignalPlot,ReconICAPlottable,Labels)
PlotReconstructionOneNumber(7, OrignalPlot,ReconICAPlottable,Labels)
#%%
PlotReconstructionMultiple(10, OrignalPlot,ReconICAPlottable,Labels)
#%%
PlotComponents(desired_dim, A)
#%%
ChangeComponents(desired_dim, 1, N, ica)

#%%
ErrorDifferentDimens(1, 13, scaler, train_img, test_img, int(datasplit*N), OrignalPlot)


#%% Only 7's
Number = 7
train_img7 ,test_img7 ,OrignalPlot7 ,scaler7 , N7 , D7, SplitPoint7  =  loaddataOneNumber(mnist_dataset, datasplit, 7)
ica7,SourceICA7, ReconICAPlottable7, A7 = ApplyICA(desired_dim, scaler7, train_img7, test_img7, N7-SplitPoint7)

MSErrors, MErrors = ErrorDifferentDimens(1, 13, scaler7, train_img7, test_img7, int(datasplit*N7)+1, OrignalPlot7)


#%%
# PlotReconstructionICAOneN(20, OrignalPlot7,ReconICAPlottable7,7)

# PlotComponents(desired_dim, A7)
values = [-0.02, -0.015, -0.01, -0.005]
values2 = [ 0 ,0.005 ,0.01 ,0.015]
ChangeComponents(desired_dim, 1,len(values), N7, ica7, values,scaler7)
ChangeComponents(desired_dim, 1,len(values2), N7, ica7, values2,scaler7)

asd = ChangingComponent(desired_dim, 1, N7, ica7, -0.01,scaler7)
#%%


#%%
def reconstruct(latentspace):
    recon_ica_test_comp0 = ica7.inverse_transform([latentspace])
    recon_ica_test_comp0 = scaler7.inverse_transform(recon_ica_test_comp0)
    rec_ica_test_plottable_comp0 = np.reshape(recon_ica_test_comp0,(28,28))
    return rec_ica_test_plottable_comp0


def update(val):
    z_arr = np.array([z1_slider.val, z2_slider.val, z3_slider.val, z4_slider.val], dtype=np.single)
    plot.set_data(reconstruct(z_arr))
    fig.canvas.draw_idle()


#%% Slider
I = 0


fig = plt.figure()
plot = plt.imshow(ReconICAPlottable7[I], cmap=plt.get_cmap('gray'))
plt.subplots_adjust(bottom=0.4)

z_arr = np.zeros(4)

ax_z1 = plt.axes([0.25, 0.1, 0.65, 0.03])
z1_slider = Slider(ax=ax_z1, label='Z1', valmin=-0.05, valmax=0.05, valinit=z_arr[0])
ax_z2 = plt.axes([0.25, 0.15, 0.65, 0.03])
z2_slider = Slider(ax=ax_z2, label='Z2', valmin=-0.05, valmax=0.05, valinit=z_arr[1])
ax_z3 = plt.axes([0.25, 0.2, 0.65, 0.03])
z3_slider = Slider(ax=ax_z3, label='Z3', valmin=-0.05, valmax=0.05, valinit=z_arr[2])
ax_z4 = plt.axes([0.25, 0.25, 0.65, 0.03])
z4_slider = Slider(ax=ax_z4, label='Z4', valmin=-0.05, valmax=0.05, valinit=z_arr[3])
    
z1_slider.on_changed(update)
z2_slider.on_changed(update)
z3_slider.on_changed(update)
z4_slider.on_changed(update)

#%%
ComponentToChange = 4
NumberOfSteps = 400
start = -0.05
end = 0.05


a1 = np.arange(0,start, -(end-start)/(NumberOfSteps/4))
a2 = np.arange(start,end, (end-start)/(NumberOfSteps/4))
a3 = np.arange(end,0, -(end-start)/(NumberOfSteps/4))
a0 = np.array([0])
a = np.concatenate([a1,a2,a3,a0])

plt.ion()
fig = plt.figure(3)
plt.clf()
t = -0.05
u = np.copy(ChangingComponent(desired_dim, ComponentToChange, N7, ica7, t,scaler7))

#figure initialization
img = plt.imshow(u, cmap=plt.get_cmap('gray'))
tlt = plt.title("Influence on Component "+ str(ComponentToChange) + ": "+str(np.round(t,2)))

count = 0

def animate(frame):
    global t, u, count
    t = a[count]
    count += 1
    u = np.copy(ChangingComponent(desired_dim, ComponentToChange, N7, ica7, t, scaler7))
    img.set_array(u)
    tlt.set_text("Influence on Component "+ str(ComponentToChange) + ": "+str(np.round(t,2)))
    return img

anim = animation.FuncAnimation(fig,animate,NumberOfSteps,interval=10,repeat=False)


filename = "C:/Users/aglas/OneDrive/Bureaublad/Documenten/TW Jaar 3/CSE Minor/Final Minor Project/Mnist/Videos Slider/Animation2-ICA-C" +str(ComponentToChange)+ ".gif"

# DeleteFile(filename)

writergif = animation.PillowWriter(fps=30) 
anim.save(filename, writer=writergif)
