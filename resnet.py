#%%
# from google.colab import drive
# drive.mount('\content\drive')

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
import datetime
import pandas as pd
#%%
# model from How to Train an Image Classifier in PyTorch and use it to Perform Basic Inference on Single Images
#freezing trained layers

class Resnet_model():
    def __init__(self, epochs, drive=True):
        #initializing model
        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Dropout(0.2),
                                nn.Linear(512, 10),
                                nn.LogSoftmax(dim=1))
        self.epochs = epochs
        if drive:
             self.model_name = 'model_resnet50_'+ datetime.datetime.now().strftime("%Y%m%d%H")

    
    def train(self, device, lr, train_data_loader, test_data_loader ):
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.fc.parameters(), lr=lr)
        self.model.to(device)

        epochs = self.epochs
        steps = 0
        running_loss = 0
        print_every = 1
        train_losses, test_losses, accuracy_test = [], [], []
        for epoch in range(epochs):
            for inputs, labels in train_data_loader:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                logps = self.model.forward(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in test_data_loader:
                            inputs, labels = inputs.to(device),labels.to(device)
                            logps = self.model.forward(inputs)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()
                            
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
                    train_losses.append(running_loss/print_every)
                    test_losses.append(test_loss/len(test_data_loader))    
                    accuracy_test.append(accuracy/len(test_data_loader))                
                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss: {running_loss/print_every:.3f}.. "
                        f"Test loss: {test_loss/len(test_data_loader):.3f}.. "
                        f"Test accuracy: {accuracy/len(test_data_loader):.3f}")
                    running_loss = 0
                    self.model.train()
                plt.plot(train_losses, label='Training loss')
                plt.plot(test_losses, label='Validation loss')
                plt.show()
                plt.savefig('./drive/MyDrive/models/Resnet_Loss_Graph_5_epochs.jpg')
                df = pd.DataFrame(list(zip(train_losses, test_losses, accuracy_test)),
                columns =['train_losses', 'test_losses','accuracy_test'])
                df.to_csv(str(self.model_name)+'.csv')

    

    def save(self):
        torch.save(self.model.state_dict(),str(self.model_name)+'.pt')


 
# %%
if __name__ =='main':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)



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
    test_data = datasets.ImageFolder(root='data\FIRE-SMOKE-DATASET\Test', transform=transforms_test)#drive/MyDrive/
    print(len(train_data), len(test_data))


    #spliting the data
    train_data, val_data = torch.utils.data.random_split(train_data, [2400,300])

    #loading the data
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=64, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data,  batch_size=64, shuffle=True)

    print (len(train_data_loader),len(test_data_loader))

# %%
