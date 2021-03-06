import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from matplotlib.widgets import Slider, Button

from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from data import DisplacementDataset

import meshio


# In[2]:


#Select GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image transformations
#transform = transforms.Compose([transforms.ToTensor(),])

batch_size = 128

mean = -0.0002
std = 0.0008
normalize_scaling_factor = 0.0035292476 + 0.00306241
offset = -0.00306241

training_data = DisplacementDataset(
   data_folder="../../Dataset/Data_nonlinear_new//",
   train=True,
   ratio=0.2,
   size=0.1,
   seed=0,
   tensor=True,
   scaling_factor=normalize_scaling_factor,
   offset=offset
)

validation_data = DisplacementDataset(
   data_folder="../../Dataset/Data_nonlinear_new//",
   train=False,
   ratio=0.2,
   size=0.1,
   seed=0,
   tensor=True,
   scaling_factor=normalize_scaling_factor,
   offset=offset
)

# Create data loaders.
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

print(len(training_data))
print(len(validation_data))

checking_data = next(iter(val_loader))
print(f"Feature batch shape: {checking_data.size()}")
print(checking_data)

# In[28]:

#size of layers


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(3272, l1)
        #self.linear2 = nn.Linear(l1, l2)
        self.linear3 = nn.Linear(l1, latent_dimensions)
        self.linear4 = nn.Linear(l1, latent_dimensions)


        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        #x = torch.flatten(x, start_dim=1)
        #x = self.linear1(x)
        x = torch.sigmoid(self.linear1(x))

        #reparametrization
        mu =  (self.linear3(x))
        log_var = (self.linear4(x))
        sigma = torch.exp(log_var / 2)

        q = torch.distributions.Normal(mu, sigma)
        z = q.rsample()

        #KL-term
        self.kl = -0.5 * torch.sum(1 + log_var - mu*mu - sigma*sigma)
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dimensions, l1)
        #self.linear2 = nn.Linear(l2, l1)
        self.linear3 = nn.Linear(l1, 3272)

    def forward(self, z):
        #z = torch.sigmoid(self.linear1(z))
        z = torch.sigmoid(self.linear1(z))
        z = (self.linear3(z))


        return z

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dimensions)
        self.decoder = Decoder(latent_dimensions)

    def forward(self, x):
        x.to(device)
        z = self.encoder(x)
        return self.decoder(z)


# In[29]:


def train(model, dataloader, optimizer, lamda):
    model.train()
    running_loss = 0
    MSE_tot = 0
    KLD_tot = 0

    for data in dataloader:
        #data, _ = data
        data = data.to(device)
        data_hat = model(data)

        MSE = ((data-data_hat)**2).sum()
        KLD = model.encoder.kl
        loss = MSE + (lamda)*KLD

        running_loss += loss.item()
        KLD_tot += KLD
        MSE_tot += MSE

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss_1 = running_loss/len(dataloader.dataset)
    MSE_avg = MSE_tot/len(dataloader.dataset)
    KLD_avg = KLD_tot/len(dataloader.dataset)

    return train_loss_1, MSE_avg, KLD_avg

def validate(model, dataloader, lamda):
    model.eval()
    val_loss = 0.0
    MSE_tot = 0
    KLD_tot = 0
    with torch.no_grad():
        for data in dataloader:
            #data, _ = data
            data = data.to(device)

            #encoded_data = model.encoder(data)
            data_hat = model(data)

            MSE = ((data-data_hat)**2).sum()
            KLD = model.encoder.kl
            loss = MSE +(lamda)* KLD

            KLD_tot += KLD
            MSE_tot += MSE

            val_loss += loss.item()

    val_loss_1 = val_loss/len(dataloader.dataset)
    MSE_avg = MSE_tot/len(dataloader.dataset)
    KLD_avg = KLD_tot/len(dataloader.dataset)

    return val_loss_1, MSE_avg, KLD_avg


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()



#layer sizes
l1 = 2250
l2 = 750
latent_dimensions = 4
n = 1

lamda = 0.001


VAE = VariationalAutoencoder(latent_dimensions).to(device)
learning_rate = 0.0001
optimizer = optim.Adam(VAE.parameters(), lr=learning_rate)


print("--------------------------")
print("Latent dimensions = ", latent_dimensions)
print("--------------------------\n")

retrain = input("Retrain the model? (y/n)\n")

if retrain == "y":
    if not os.path.isdir("./Models_test"):
        os.mkdir("./Models_test")
    if not os.path.isdir("./Loss_txt"):
        os.mkdir("./Loss_txt")
    if not os.path.isdir("./Figures"):
        os.mkdir("./Figures")

if (not os.path.isfile("./Models_test/VAE-{}-{}-{}_{}.pth".format(latent_dimensions, l1, l2, lamda))) or retrain == "y":
    num_epochs = 30

    for epoch in range(num_epochs):
        train_loss, MSE_t, KL_t = train(VAE, train_loader, optimizer, lamda)
        val_loss, MSE_v, KL_v = validate(VAE, val_loader, lamda)
        print("epoch = {}, train_loss = {:.6f}, val_loss = {:.6f}".format(epoch, train_loss, val_loss))

    torch.save(VAE.state_dict(), "./Models_test/VAE-{}-{}-{}_{}.pth".format(latent_dimensions, l1, l2, lamda))

    print('\n lamda = {} \t train loss {:.6f} \t val loss {:.6f} '.format(lamda, train_loss, val_loss))
    print(' lamda = {} \t MSE train {:.6f} \t KL train {:.6f} '.format(lamda, MSE_t, KL_t))
    print(' lamda = {} \t MSE val {:.6f} \t KL val {:.6f} '.format(lamda, MSE_v, KL_v))

    Recon_loss_t = MSE_t
    KL_loss_t = KL_t

    Recon_loss_v = MSE_v
    KL_loss_v = KL_v

    #resetting weights of Neural network
    VAE.apply(weight_reset)

    #np.savetxt("Loss_txt/{}-{}_{}.txt".format(latent_dimensions, l1, lamda), (lamda, Recon_loss_t, Recon_loss_v, KL_loss_t, KL_loss_v), fmt = "%1.6f", delimiter = ", ")

else:
    VAE.load_state_dict(torch.load("./Models_test/VAE-{}-{}-{}_{}.pth".format(latent_dimensions, l1, l2, lamda)))
    print("./Models_test/VAE-{}-{}-{}_{}.pth".format(latent_dimensions, l1, l2, lamda))


VAE.load_state_dict(torch.load("./Models_test/VAE-{}-{}-{}_{}.pth".format(latent_dimensions, l1, l2, lamda)))




N=5

def plot_rec(encoder,decoder,N):
    plt.figure(figsize = (8,4.5))
    for i in range(N):
        ax = plt.subplot(2,N,i+1)
        img = validation_data[i+np.random.randint(0, 1020)][0].to(device)
        plt.imshow(img.squeeze().numpy(), cmap='gist_gray')


        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z = encoder(img)
            print(z)
            rec_img  = decoder(z)

        ax = plt.subplot(2, N, i + 1 + N)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')

    plt.show()



def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
    plt.show()


def GetPointsAndCells():
    filename  = "../../DataSet/rveLinearMultiple/para_1.vtu"
    mesh = meshio.read(filename)
    points, cells = mesh.points, mesh.cells
    return points, cells

def create_vtu(para,filename):
    Disp = np.reshape(para,(1636,2))
    Disp_xyz = np.zeros((Disp.shape[0],Disp.shape[1]+1))
    Disp_xyz[:,:-1] = Disp
    point_data = {"Displacement":Disp_xyz}
    MakeVTUFile(points,cells,point_data, {}, filename)

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

points, cells= GetPointsAndCells()

def VAE_VTU(data):
    VAE.encoder.eval()
    VAE.decoder.eval()
    with torch.no_grad():
        z = VAE.encoder(data)
        z_arr = z.to("cpu").detach().numpy()
        print(z_arr)
        result  = VAE.decoder(z)
        result = result.to("cpu").detach().numpy()

        np.savetxt("test_data_input.txt", data)
        np.savetxt("test_data_output.txt", result)

        create_vtu(data, "input_test.vtu")
        create_vtu(result, "test.vtu")

VAE_VTU(checking_data[0])




# edit_number = input("Edit number with latent dimensions? (y/n)\n")
# if edit_number == "y":

#     img = validation_data[np.random.randint(0, 1020)][0].to(device)

#     VAE.encoder.eval()
#     VAE.decoder.eval()

#     z_arr = np.zeros(latent_dimensions)

#     with torch.no_grad():
#         z = VAE.encoder(img)
#         z_arr = z.to("cpu").detach().numpy()
#         rec_img  = VAE.decoder(z)

#     fig = plt.figure()
#     plot = plt.imshow(rec_img.squeeze().numpy(), cmap='gist_gray')

#     plt.subplots_adjust(bottom=0.4)

#     #for n in range(latent_dimensions):

#     z_arr = z_arr[0]

#     ax_z1 = plt.axes([0.25, 0.1, 0.65, 0.03])
#     z1_slider = Slider(ax=ax_z1, label='Z1', valmin=-20, valmax=20, valinit=z_arr[0])
#     ax_z2 = plt.axes([0.25, 0.15, 0.65, 0.03])
#     z2_slider = Slider(ax=ax_z2, label='Z2', valmin=-20, valmax=20, valinit=z_arr[1])
#     ax_z3 = plt.axes([0.25, 0.2, 0.65, 0.03])
#     z3_slider = Slider(ax=ax_z3, label='Z3', valmin=-20, valmax=20, valinit=z_arr[2])
#     ax_z4 = plt.axes([0.25, 0.25, 0.65, 0.03])
#     z4_slider = Slider(ax=ax_z4, label='Z4', valmin=-20, valmax=20, valinit=z_arr[3])

#     def update_plot(val):
#         z_arr = np.array([z1_slider.val, z2_slider.val, z3_slider.val, z4_slider.val], dtype=np.single)
#         z = torch.from_numpy(z_arr)
#         VAE.decoder.eval()
#         with torch.no_grad():
#             rec_img = VAE.decoder(z)
#         plot.set_data(rec_img.squeeze().numpy())
#         fig.canvas.draw_idle()

#     z1_slider.on_changed(update_plot)
#     z2_slider.on_changed(update_plot)
#     z3_slider.on_changed(update_plot)
#     z4_slider.on_changed(update_plot)

#     #resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
#     #button = Button(resetax, 'Reset', hovercolor='0.975')

#     plt.show()


def get_z(X, Y):
    data = np.loadtxt("Loss_txt/Recon_loss_{}-{}_sigmoid.txt".format(X, l1), usecols=1, delimiter = ", ")
    index = int(np.log10(Y)+6)
    return data[index]

def surface_plot():
    Y, X = lamda_seq, [2, 4, 8]
    Z = np.zeros((len(X), len(Y)))

    count = 0
    for x in [2, 4, 8]:
        for y in lamda_seq:
            Z[count // len(Y), count % len(Y)] = get_z(x, y)
            count += 1
    X = np.transpose(np.tile(X, (len(Y), 1)))
    Y = np.log10(np.tile(Y, (len(X), 1)))
    ax = plt.axes(projection="3d")
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=0)
    ax.set_xlabel("Latent Dimensions")
    ax.set_ylabel("Lambda (log)")
    ax.set_zlabel("Reconstruction error")
    plt.show()

#surface_plot()
