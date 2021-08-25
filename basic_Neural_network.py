import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from numpy import isnan
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sklearn
from torch.nn import functional as F
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from os import path as os_path
from torch.utils.data import DataLoader
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.metrics import f1_score
import torch.optim as optim
#%%
transforms_train = transforms.Compose([transforms.Resize(225),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

transforms_test = transforms.Compose([transforms.Resize(225),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5],
                                                            [0.5, 0.5, 0.5])])

train_data = datasets.ImageFolder(root='data\FIRE-SMOKE-DATASET\Train', transform=transforms_train)
test_data = datasets.ImageFolder(root='data\FIRE-SMOKE-DATASET\Test', transform=transforms_test)
print(len(train_data), len(test_data))

#%%
#spliting the data
train_data, val_data = torch.utils.data.random_split(train_data, [2400,300])
#%%
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data,  batch_size=64, shuffle=True)


#%%
device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")

#%%

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(43008,512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,50),
            torch.nn.ReLU(),
            torch.nn.Linear(50,10),
            torch.nn.LogSoftmax(dim=1)
            #Linear(2048, 512),
            #ReLU(),
            #Dropout(0.2),
            #Linear(512, 10),
            #LogSoftmax(dim=1))
        )
    def forward(self, X):
        return self.layers(X)

net = NeuralNetwork()

def train_nn(model, train_dataloader, epochs=10, lr=0.01, print_losses=True):
    # for param in model.parameters():
    #     print(f"Parameters are: {param}")
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # create optimiser
    loss = torch.nn.CrossEntropyLoss()
    losses = []
    scores = []
    for epoch in range(epochs):
        for X,y in train_dataloader:
            optimizer.step()
            optimizer.zero_grad()
            y_hat = net(X)
            y_hat = torch.squeeze(y_hat)
            print('y_hat shape:',y_hat.shape)
            print('y shape:',y.shape)
            loss =loss(y_hat,y)  # this only works if the target is dim-1 - should use n_labels
            if print_losses:
                print(f"loss:{loss}")
            loss.backward()  # Upgrades the .grad -- of each of the parameters (based on backpopulating through the NN)
            losses.append(loss.item())
            r2_score = sklearn.metrics.r2_score(y_hat.detach().numpy(), y.detach().numpy())
            scores.append(r2_score)
    fig, axs = plt.subplots(2)
    axs[0].plot(losses)
    axs[1].plot(scores)
    fig.suptitle('vertically stacked subplots' )
    plt.show()
#%%
train_nn(net, train_data_loader, epochs=5, lr=0.000001, print_losses=True)
