#!/usr/bin/env python
# coding: utf-8




# import packages
import os
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
 
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image


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

# Create data loaders.
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=False)

print(len(train_loader))

checking_data, n_o_i = next(iter(train_loader))
print(f"Feature batch shape: {checking_data.size()}")





#VAE encoder and decoder

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dimensions)
        self.linear3 = nn.Linear(512, latent_dimensions)

        self.N = torch.distributions.Normal(0, 1)
        #self.N.loc = self.N.loc.cuda() # hack to get sampling on the GPU
        #self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        
        #reparametrization
        mu =  self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        
        #KL-term
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dimensions, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        
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
    


def train(model, dataloader, optimizer):
    model.train()
    running_loss = 0
        
    for data in dataloader:
        data, _ = data
        data = data.to(device)
        data_hat = model(data)
        
        loss = ((data-data_hat)**2).sum() + (lamda/2)*model.encoder.kl
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
         
    train_loss_1 = running_loss/len(dataloader.dataset)

    return train_loss_1, model

def validate(model, dataloader):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data in dataloader:
            data, _ = data
            data = data.to(device)
            
            encoded_data = model.encoder(data)
            data_hat = model(data)
            
            loss = ((data-data_hat)**2).sum() + (lamda/2)*model.encoder.kl
            val_loss += loss.item()
                
    val_loss_1 = val_loss/len(dataloader.dataset)
    
    return val_loss_1


latent_dimensions = 4
lamda = 0.0001

VAE = VariationalAutoencoder(latent_dimensions).to(device)

learning_rate = 0.0001

optimizer = optim.Adam(VAE.parameters(), lr=learning_rate, weight_decay=1e-5)

retrain = input("Retrain the model? (y/n)\n")

if (not os.path.isfile("./model.pth")) or retrain == "y":
    num_epochs = 10
    for epoch in range(num_epochs):
        train_loss, VAE = train(VAE, train_loader, optimizer)
        val_loss = validate(VAE, val_loader)
        torch.save(VAE.state_dict(), "./model.pth")
        print('\n EPOCH {}/{} \t train loss {:.3f} \t val loss {:.3f}'.format(epoch + 1, num_epochs,train_loss,val_loss))
else:
    VAE.load_state_dict(torch.load("./model.pth"))

n = 5

def plot_rec(encoder,decoder,n):
    plt.figure(figsize = (8,4.5))
    for i in range(n):
        ax = plt.subplot(2,n,i+1)
        img = validation_data[i+np.random.randint(0, 778)][0].to(device)
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


def plot_latent(autoencoder, data, num_batches=100):
    for i, (x, y) in enumerate(data):
        z = autoencoder.encoder(x.to(device))
        z = z.to('cpu').detach().numpy()
        plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
        if i > num_batches:
            plt.colorbar()
            break
            
plot_latent(VAE, train_loader)





#def plot_reconstructed(decoder, r0=(-5, 10), r1=(-10, 5), n2=10):
#    w = 28
#    img = np.zeros((n2*w, n2*w))
#    for i, y in enumerate(np.linspace(*r1, n2)):
#        for j, x in enumerate(np.linspace(*r0, n2)):
#           z = torch.Tensor([[x, y]]).to(device)
#            x_hat = decoder(z)
#            x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
#           img[(n2-1-i)*w:(n2-1-i+1)*w, j*w:(j+1)*w] = x_hat
#    plt.imshow(img, extent=[*r0, *r1])
    
    
#plot_reconstructed(VAE.decoder, r0=(-2, 2), r1=(-2, 2))