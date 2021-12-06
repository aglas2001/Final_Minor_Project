#!/usr/bin/env python
# coding: utf-8

# In[4]:


# import packages
import os
import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
 
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

#Import custom dataloader & kfold tool
import data
from sklearn.model_selection import KFold

#Select GPU if available 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 64

# Create data loaders.
training_data = data.DisplacementDataset("../test_data/para_1.vtu")
validation_data = data.DisplacementDataset("../test_data/para_1.vtu")
dataset = torch.utils.data.ConcatDataset([training_data, validation_data])


#Changed input & output to 1034 nodes (size of dataset)
class LinearVAE(nn.Module):
    def __init__(self):
        super(LinearVAE, self).__init__()
 
        # encoder
        self.enc1 = nn.Linear(in_features=1034, out_features=512)
        self.enc2 = nn.Linear(in_features=512, out_features=latent_dimensions*2)
 
        # decoder 
        self.dec1 = nn.Linear(in_features=latent_dimensions, out_features=512)
        self.dec2 = nn.Linear(in_features=512, out_features=1034)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling as if coming from the input space
        return sample
 
    def forward(self, x):
        # encoding
        x = F.relu(self.enc1(x))
        x = self.enc2(x).view(-1, 2, latent_dimensions)
        
        # get mu and log_var
        mu = x[:, 0, :] # the first feature values as mean
        log_var = x[:, 1, :] # the other feature values as variance
        
        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
 
        # decoding
        x = F.relu(self.dec1(z))
        reconstruction = torch.sigmoid(self.dec2(x))
        
        return reconstruction, mu, log_var


#Final loss function 
def final_loss(bce_loss, mu, logvar):
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(model, dataloader, epochs):
    train_loss = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0
        
        for data in train_loader:
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            optimizer.zero_grad()
        
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()

        
        train_loss_1 = running_loss/len(dataloader.dataset)
        train_loss.append(train_loss_1)
        
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(epoch+1, epochs, train_loss_1))
        
        img = data.view(data.size(0), 1, 28, 28)
        save_image(img, './MNIST_VAE/linear_ae_image{}.png'.format(epoch))
        
    return train_loss


def validate(model, dataloader, epochs):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(validation_data)/dataloader.batch_size)):
            data, _ = data
            data = data.to(device)
            data = data.view(data.size(0), -1)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
        
            # save the last batch input and output of every epoch
            #if i == int(len(val_data)/dataloader.batch_size) - 1:
             #   num_rows = 8
              #  both = torch.cat((data.view(batch_size, 1, 28, 28)[:8], 
               #                   reconstruction.view(batch_size, 1, 28, 28)[:8]))
                #save_image(both.cpu(), f"../outputs/output{epoch}.png", nrow=num_rows)
        
                
    val_loss_1 = running_loss/len(dataloader.dataset)
    print('After {:d} epochs the Validation Loss: {:.3f}'.format( epochs, val_loss_1))
    
    return val_loss_1

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

# define a simple linear VAE
latent_dimensions = 16

#learning parameters for VAE
folds = 4
epochs = 10
batch_size = 64
learning_rate = 0.0001

#Initialize the KFold
kfold = KFold(n_splits=folds, shuffle=True)

# Start print
print('--------------------------------')

# K-fold Cross Validation model evaluation
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(
                      dataset, 
                      batch_size=10, sampler=train_subsampler)
    test_loader = DataLoader(
                      dataset,
                      batch_size=10, sampler=test_subsampler)
    
    # Init the neural network
    model = LinearVAE().to(device)
    model.apply(reset_weights)
    
    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)

    criterion = nn.BCELoss(reduction='sum')

    train_loss = train(model, train_loader, epochs)
            
    # Process is complete.
    print('Training process has finished. Saving trained model.')

    # Print about testing
    print('Starting testing')
    
    # Saving the model
    save_path = f'./model-fold-{fold}.pth'
    torch.save(model.state_dict(), save_path)

    # Evaluation for this fold
    val_loss.append(val_epoch_loss)
    print("Validation Loss = %.3f" % (val_epoch_loss))

    val_epoch_loss = validate(model, val_loader, epochs)
    
# Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
