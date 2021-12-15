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


# In[2]:


#Select GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image transformations
transform = transforms.Compose([transforms.ToTensor(),])

batch_size = 128

#Loading DATA
training_data = datasets.MNIST(
    root="data_train",
    train=True,
    download=True,
    transform=transform,
)

# Download test data from open datasets.
validation_data = datasets.MNIST(
    root="data_val",
    train=False,
    download=True,
    transform=transform,
)

label_indices_training = training_data.targets == 7
label_indices_validation = validation_data.targets == 7
training_data.data, training_data.targets = training_data.data[label_indices_training], training_data.targets[label_indices_training]
validation_data.data, validation_data.targets = validation_data.data[label_indices_validation], validation_data.targets[label_indices_validation]

# Create data loaders.
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

print(len(training_data))
print(len(validation_data))

checking_data, n_o_i = next(iter(train_loader))
print(f"Feature batch shape: {checking_data.size()}")


# In[28]:


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, latent_dimensions)
        self.linear4 = nn.Linear(128, latent_dimensions)

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = torch.sigmoid(self.linear1(x))
        x = torch.tanh(self.linear2(x))

        #reparametrization
        mu =  (self.linear3(x))
        log_var = (self.linear4(x))
        sigma = torch.exp(log_var / 2)

        q = torch.distributions.Normal(mu, sigma)
        z = q.rsample()

        #KL-term
        self.kl = -0.5 * torch.sum(1 + log_var - mu*mu - log_var.exp())
        return z

class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dimensions, 128)
        self.linear2 = nn.Linear(128, 512)
        self.linear3 = nn.Linear(512, 784)

    def forward(self, z):
        z = torch.relu(self.linear1(z))
        z = torch.tanh(self.linear2(z))
        z = torch.sigmoid(self.linear3(z))

        return z.reshape((-1, 1, 28, 28))

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
        data, _ = data
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
            data, _ = data
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


# In[ ]:





# In[30]:


latent_dimensions = 4

n = 8
lamda_seq = np.logspace(-6, 1, n)

Recon_loss_t = np.zeros(n)
KL_loss_t = np.zeros(n)

Recon_loss_v = np.zeros(n)
KL_loss_v = np.zeros(n)

VAE = VariationalAutoencoder(latent_dimensions).to(device)
learning_rate = 0.0001
optimizer = optim.Adam(VAE.parameters(), lr=learning_rate)

retrain = input("Retrain the model? (y/n)\n")

if (not os.path.isfile("./saved_models/model_lamda_complex_0.pth")) or retrain == "y":
    num_epochs = 10

    for i in range(n):
        lamda = lamda_seq[i]
        for epoch in range(num_epochs):
            train_loss, MSE_t, KL_t = train(VAE, train_loader, optimizer, lamda)
            val_loss, MSE_v, KL_v = validate(VAE, val_loader, lamda)
            torch.save(VAE.state_dict(), "./saved_models/model_lamda_complex_{}.pth".format(i))


        print('\n lamda = {} \t train loss {:.3f} \t val loss {:.3f} '.format(lamda,train_loss,val_loss))
        print(' lamda = {} \t MSE train {:.3f} \t KL train {:.3f} '.format(lamda,MSE_t, KL_t))
        print(' lamda = {} \t MSE val {:.3f} \t KL val {:.3f} '.format(lamda,MSE_v, KL_v))

        Recon_loss_t[i] = MSE_t
        KL_loss_t[i] = KL_t

        Recon_loss_v[i] = MSE_v
        KL_loss_v[i] = KL_v
        
    np.savetxt("Recon_loss.txt", np.transpose([Recon_loss_t, Recon_loss_v]), delimiter = ",")
    np.savetxt("KL_loss.txt",  np.transpose([KL_loss_t, KL_loss_v]), delimiter = ",")

else:
    VAE.load_state_dict(torch.load("./saved_models/model_lamda_complex_4.pth"))


# In[59]:


plt.figure(figsize = (10,8))
plt.subplot(1,2,1)
plt.plot(lamda_seq, KL_loss_t, label = "KL for training data")
plt.plot(lamda_seq, KL_loss_v, label = "KL for validation data")
plt.xscale("log")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(lamda_seq, Recon_loss_t, label = "Reconstruction loss training data")
plt.plot(lamda_seq, Recon_loss_v, label = "Reconstruction loss validation data")
plt.xscale("log")
plt.legend()
plt.grid()

plt.savefig("Loss_plot_complex_non_sym.png")
plt.show()


# In[47]:


VAE.load_state_dict(torch.load("./saved_models/model_lamda_complex_5.pth"))


# In[53]:


n=5

def plot_rec(encoder,decoder,n):
    plt.figure(figsize = (8,4.5))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        img = validation_data[i+np.random.randint(0, 1020)][0].to(device)
        plt.imshow(img.squeeze().numpy(), cmap='gist_gray')


        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            z = encoder(img)
            print(z)
            rec_img  = decoder(z)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')

    plt.show()

plot_rec(VAE.encoder, VAE.decoder, n)


# In[54]:


def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break

plot_latent(VAE, train_loader)


# In[ ]:


#print(train_loss)


# In[36]:



edit_number = input("Edit number with latent dimensions? (y/n)\n")

if edit_number == "y":

    img = validation_data[np.random.randint(0, 1020)][0].to(device)

    VAE.encoder.eval()
    VAE.decoder.eval()

    z_arr = np.zeros(latent_dimensions)

    with torch.no_grad():
        z = VAE.encoder(img)
        z_arr = z.to("cpu").detach().numpy()
        rec_img  = VAE.decoder(z)

    fig = plt.figure()
    plot = plt.imshow(rec_img.squeeze().numpy(), cmap='gist_gray')

    plt.subplots_adjust(bottom=0.4)

    #for n in range(latent_dimensions):

    z_arr = z_arr[0]

    ax_z1 = plt.axes([0.25, 0.1, 0.65, 0.03])
    z1_slider = Slider(ax=ax_z1, label='Z1', valmin=-20, valmax=20, valinit=z_arr[0])
    ax_z2 = plt.axes([0.25, 0.15, 0.65, 0.03])
    z2_slider = Slider(ax=ax_z2, label='Z2', valmin=-20, valmax=20, valinit=z_arr[1])
    ax_z3 = plt.axes([0.25, 0.2, 0.65, 0.03])
    z3_slider = Slider(ax=ax_z3, label='Z3', valmin=-20, valmax=20, valinit=z_arr[2])
    ax_z4 = plt.axes([0.25, 0.25, 0.65, 0.03])
    z4_slider = Slider(ax=ax_z4, label='Z4', valmin=-20, valmax=20, valinit=z_arr[3])

    def update_plot(val):
        z_arr = np.array([z1_slider.val, z2_slider.val, z3_slider.val, z4_slider.val], dtype=np.single)
        z = torch.from_numpy(z_arr)
        VAE.decoder.eval()
        with torch.no_grad():
            rec_img = VAE.decoder(z)
        plot.set_data(rec_img.squeeze().numpy())
        fig.canvas.draw_idle()

    z1_slider.on_changed(update_plot)
    z2_slider.on_changed(update_plot)
    z3_slider.on_changed(update_plot)
    z4_slider.on_changed(update_plot)

    #resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
    #button = Button(resetax, 'Reset', hovercolor='0.975')

    plt.show()
