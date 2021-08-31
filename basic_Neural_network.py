#%%
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

#spliting the data
train_data, val_data = torch.utils.data.random_split(train_data, [2400,300])

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data,  batch_size=64, shuffle=True)

classes =('Fire', 'Neutral', 'Smoke')

dataiter = iter(train_data_loader)
images, labels = dataiter.next()
print(labels)
classes[labels[0]]

device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(224,224),
            torch.nn.ReLU(),
            torch.nn.Linear(224,50),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(33600,500),
            torch.nn.ReLU(),
            torch.nn.Linear(500,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,3),
            torch.nn.LogSoftmax(dim=1)
        )
    def forward(self, X):
        return self.layers(X)

net = NeuralNetwork()

def train_nn(model, train_dataloader, epochs=10, lr=0.01, print_losses=True):
    # for param in model.parameters():
    #     print(f"Parameters are: {param}")
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # create optimiser
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    scores = []
    for epoch in range(epochs):
        for X,y in train_dataloader:
            optimizer.zero_grad()
            y_hat = net(X)
            #y_hat = torch.argmax(y_hat, dim=1)
            #y_hat = torch.squeeze(y_hat)
            loss =criterion(y_hat,y)  # this only works if the target is dim-1 - should use n_labels
            loss.backward()  # Upgrades the .grad -- of each of the parameters (based on backpopulating through the NN)
            optimizer.step()
            losses.append(loss.item())
            ps = torch.exp(y_hat) #Returns a new tensor with the exponential of the elements of the input tensor
            top_p, top_class = ps.topk(1, dim=1)#Returns the k largest elements of the given input tensor along a given dimension
            f1_score = sklearn.metrics.f1_score(top_class.detach().numpy(), y.detach().numpy())
            if print_losses:
                print(f"loss:{loss}")
                print(f"score:{f1_score}")
            scores.append(f1_score)
    fig, axs = plt.subplots(2)
    axs[0].plot(losses)
    axs[1].plot(scores)
    fig.suptitle('losses/scores' )
    plt.show()
    plt.savefig('basic_neural_network_10_epochs.png')
    df = pd.DataFrame(list(zip(losses, scores)),
               columns =['losses', 'scores'])
    df.to_csv('basic_Neural_network_Loss_10_epochs')

train_nn(net, train_data_loader, epochs=10, lr=0.01, print_losses=True)

# score:0.08133971291866027
# loss:0.8883010745048523
# score:0.06394557823129254
# loss:0.7468932867050171
# score:0.4863563402889246

# %%

# main errors were related to flatten and convert y_hat to top_class
# %%
