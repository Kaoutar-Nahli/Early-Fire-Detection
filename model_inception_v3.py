#%%
#from google.colab import drive
#drive.mount('\content\drive')
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


                                       
transforms_test = transforms.Compose([transforms.Resize((299,299)),
                                       #transforms.CenterCrop(299),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transforms_train = transforms.Compose([transforms.Resize((299,299)),
                                       transforms.AutoAugment(),
                                       #transforms.CenterCrop(229),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


train_data = datasets.ImageFolder(root='data/FIRE-SMOKE-DATASET/Train', transform=transforms_train) #drive/MyDrive/
test_data = datasets.ImageFolder(root='data/FIRE-SMOKE-DATASET/Test', transform=transforms_test) #drive/MyDrive/
print(len(train_data), len(test_data))


#spliting the data
train_data, val_data = torch.utils.data.random_split(train_data, [2400,300])

#loading the data
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data,  batch_size=64, shuffle=True)
print (train_data[0][0].size())
#%%
#initializing model
#https://pytorch.org/hub/pytorch_vision_inception_v3/

model = models.inception_v3(pretrained=True)
#print(model)
print('kkkkkk')
#%%
# model from How to Train an Image Classifier in PyTorch and use it to Perform Basic Inference on Single Images
#freezing trained layers
for param in model.parameters():
    param.requires_grad = False

for k, param in enumerate(model.parameters()):
    print (k)# 291 parameters freez first 249

#%%    
model.fc = nn.Sequential(#nn.AvgPool2d(kernel_size=(2)),
                        nn.Linear(model.fc.in_features, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 50),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(50, 3),
                        nn.LogSoftmax(dim=1))
#%%
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.fc.parameters(), lr=0.003)
model.to(device)

epochs = 2
steps = 0
running_loss = 0
print_every = 1
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in train_data_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps.logits, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_data_loader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(train_data_loader))
            test_losses.append(test_loss/len(test_data_loader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(test_data_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_data_loader):.3f}")
            running_loss = 0
            model.train()
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.show()
plt.savefig('Inception_v3_part1_20_epochs_Loss_Graph.png') #./drive/MyDrive/models/

torch.save(model.state_dict(), 'Inception_v3_part1_20_epochs.pt')#./drive/MyDrive/models/
print('kkkkkkkkkkkkkkkkkkkkkkkk')
#%%





















#%%%
# counting parameters pre children
count_layer = 0
for layer in model.children():
    count_layer +=1
    count_param = 0
    for param in layer.parameters():
        count_param +=1
    print(count_layer , count_param)

#counting children
def flatten(el):
    flattened = [flatten(children) for children in el.children()]
    res = [el]
    for c in flattened:
        res += c
    return res

layers = flatten(model)
print (len(layers))

# freeze the two top blocks, the first 249 layers
for k, param in enumerate(model.parameters()):
  
  if k < 249:
        # freeze backbone layers
        param.requires_grad = False
            
  else:
        param.requires_grad = True



# keep training model with SGD and learning rate 0.0001
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.0000001)
epochs = 2
steps = 0
running_loss = 0
print_every = 1
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in train_data_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_data_loader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(train_data_loader))
            test_losses.append(test_loss/len(test_data_loader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(test_data_loader):.3f}.. "
                  f"Test accuracy: {accuracy/len(test_data_loader):.3f}")
            running_loss = 0
            model.train()
plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.show()
plt.savefig('Inception_v3_part1_20_epochs_Loss_Graph.png') #./drive/MyDrive/models/

torch.save(model.state_dict(), 'Inception_v3_part1_20_epochs.pt')#./drive/MyDrive/models/
# %%
