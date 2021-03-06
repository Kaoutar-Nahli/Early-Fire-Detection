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
import torchvision
#%%
#pip install tensorflow
from tensorflow import summary
%load_ext tensorboard
import datetime
current_time = str(datetime.datetime.now().timestamp())
train_log_dir = 'logs/tensorboard/train/' + current_time
test_log_dir = 'logs/tensorboard/test/' + current_time
train_summary_writer = summary.create_file_writer(train_log_dir)
test_summary_writer = summary.create_file_writer(test_log_dir)

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

#spliting the data
train_data, val_data = torch.utils.data.random_split(train_data, [2400,300])

train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data,  batch_size=64, shuffle=True)

classes =('Fire', 'Neutral', 'Smoke')

dataiter = iter(train_data_loader)
images, labels = dataiter.next()
print(images.shape)
classes[labels[0]]

#%%
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

#Define Convolutional Neural Network

class Conv_NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3,6,5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(6,16,5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Flatten(),
            torch.nn.Linear(44944,500),
            torch.nn.ReLU(),
            torch.nn.Linear(500,120),
            torch.nn.ReLU(),
            torch.nn.Linear(120,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,3),
            #torch.nn.LogSoftmax(dim=1)
        )
    def forward(self, X):
        return self.layers(X)
#%%
net = Conv_NeuralNetwork()

# Define a loss function and optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) # create optimiser
criterion = torch.nn.CrossEntropyLoss()

def train_nn(model, train_dataloader, epochs=10, lr=0.01):
    train_losses = []
    test_losses = []
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for X,y in train_dataloader:
            steps+=1
            optimizer.zero_grad()
            y_hat = net(X)
            loss =criterion(y_hat,y)  # this only works if the target is dim-1 - should use n_labels
            loss.backward()  # Upgrades the .grad -- of each of the parameters (based on backpopulating through the NN)
            optimizer.step()
            # print stat
            running_loss += loss.item()
        if steps % 20 == 0:
            test_loss = 0
            accuracy = 0
            model.eval() # evaluation mode
            with torch.no_grad():
                for inputs, labels in test_data_loader:
                    #inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    ps = torch.exp(logps) #Returns a new tensor with the exponential of the elements of the input tensor
                    top_p, top_class = ps.topk(1, dim=1)#Returns the k largest elements of the given input tensor along a given dimension
                    equals = top_class == labels.view(*top_class.shape)#Returns a new tensor with the same data as the self tensor but of a different shape
                    accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(train_data_loader))# training loss per batch (38 batchses), 64 points per batch, 2700 p0ints
            test_losses.append(test_loss/len(test_data_loader)) # test loss  calculated in the 300 points in batches?                
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/20:.3f}.. " # running loss is calculated per each batch and printed every : print_every
                  f"Test loss: {test_loss/len(test_data_loader):.3f}.. "# test loss is the sum of the batch loss in the test data and divided by number of batches= 300/64
                  f"Test accuracy: {accuracy/len(test_data_loader):.3f}")
            
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.show()
    plt.savefig('Covolutional_Loss_20_epochs.png')

train_nn(net, train_data_loader, epochs=500, lr=0.01)
#%%
%tensorboard --logdir logs/tensorboard
#%%
#from datetime import datetime
#now = datetime.now()
#dt_string = f'{now.strftime("%d/%m/%Y-%H%M%S")}-basic_convolutional'

torch.save(net.state_dict(), 'basic_convolutional01.pt')
# main errors were related to flatten and convert y_hat to top_class
# %%
#loading back model
#%%
class Conv_NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3,6,5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(6,16,5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Flatten(),
            torch.nn.Linear(44944,500),
            torch.nn.ReLU(),
            torch.nn.Linear(500,120),
            torch.nn.ReLU(),
            torch.nn.Linear(120,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,3),
            #torch.nn.LogSoftmax(dim=1)
        )
    def forward(self, X):
        return self.layers(X)
#%%
net = Conv_NeuralNetwork()
net.load_state_dict(torch.load('model_basic_conv/basic_convolutional.pt'))



# %%
# let see what the neural network thinks


# %%
#Epoch 10/500.. Train loss: 19.678.. Test loss: 0.818.. Test accuracy: 0.628
#Epoch 20/500.. Train loss: 30.343.. Test loss: 0.728.. Test accuracy: 0.737


# Epoch 340/500.. Train loss: 41.124.. Test loss: 3.057.. Test accuracy: 0.657
# Epoch 350/500.. Train loss: 41.126.. Test loss: 3.024.. Test accuracy: 0.661
# %%
