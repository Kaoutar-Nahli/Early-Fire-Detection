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
import os

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

# model from How to Train an Image Classifier in PyTorch and use it to Perform Basic Inference on Single Images
class InceptionV3():
    def __init__(self, epochs=20, lr=0.01,drive=True):
        #initializing model
        #https://pytorch.org/hub/pytorch_vision_inception_v3/
        self.model = models.inception_v3(pretrained=True)
        
        #freezing trained layers
        for param in self.model.parameters():
            param.requires_grad = False

      
        self.model.fc = nn.Sequential(#nn.AvgPool2d(kernel_size=(2)),
                        nn.Linear(self.model.fc.in_features, 512),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(512, 50),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(50, 3),
                        nn.LogSoftmax(dim=1))

        self.epochs = epochs
        self.lr = lr
        if drive:
             self.model_name = './drive/MyDrive/models/model_resnet50_'+ datetime.datetime.now().strftime("%Y%m%d%H")
        self.model_name = 'models/model_resnet50_'+ str( epochs) + str(datetime.datetime.now().strftime("%Y%m%d%H"))
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=0.0000001)

    def number_parameters(self):
       list_param = []
       for param in self.model.parameters():
            list_param.append(param)# 291 parameters freez first 249
       return len(list_param)
        

    def train_part_1(self, device, train_data_loader, test_data_loader ):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.RMSprop(self.model.fc.parameters(), lr=self.lr)
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
                self.optimizer.zero_grad()
                logps = self.model.forward(inputs)
                loss = self.criterion(logps.logits, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in test_data_loader:
                            inputs, labels = inputs.to(device),labels.to(device)
                            logps = self.model.forward(inputs)
                            batch_loss = self.criterion(logps, labels)
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
                plt.savefig(str(self.model_name)+'part_1.jpg')
                df = pd.DataFrame(list(zip(train_losses, test_losses, accuracy_test)),
                columns =['train_losses', 'test_losses','accuracy_test'])
                df.to_csv(str(self.model_name)+'part_1.csv')
                self.save_dict('part_1')
    
    def train_part_2(self, device, train_data_loader, test_data_loader ):
        
        # keep training model with SGD and learning rate 0.0001
        for k, param in enumerate(self.model.parameters()):
        
            if k < 249:
                    # freeze backbone layers
                    param.requires_grad = False
                        
            else:
                    param.requires_grad = True
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=0.0000001)
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
                self.optimizer.zero_grad()
                logps = self.model.forward(inputs)
                loss = self.criterion(logps.logits, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    self.model.eval()
                    with torch.no_grad():
                        for inputs, labels in test_data_loader:
                            inputs, labels = inputs.to(device),labels.to(device)
                            logps = self.model.forward(inputs)
                            batch_loss = self.criterion(logps, labels)
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
                plt.savefig(str(self.model_name)+'part_2.jpg')
                df = pd.DataFrame(list(zip(train_losses, test_losses, accuracy_test)),
                columns =['train_losses', 'test_losses','accuracy_test'])
                df.to_csv(str(self.model_name)+'part_2.csv')
                self.save_dict('part_2')
    
    

    def save_dict(self, part):
        torch.save(self.model.state_dict(), str(self.model_name) + str(part) + '.pt')
    
    def save_all(self, part):
        state = {'epoch': self.epochs + 1, 'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict() }

        f_path = str(self.model_name) + self(part) + '.pt'
        #os.path.join(checkpoint_dir, 'checkpoint.pth') 

        torch.save(state, f_path)



#if loading model is required after loading model to device
#model = model.to(device) after loading model trained previously in different parameters
# counting parameters pre children
# count_layer = 0
# for layer in model.children():
#     count_layer +=1
#     count_param = 0
#     for param in layer.parameters():
#         count_param +=1
#     print(count_layer , count_param)

#counting children
# def flatten(el):
#     flattened = [flatten(children) for children in el.children()]
#     res = [el]
#     for c in flattened:
#         res += c
#     return res

# layers = flatten(model)
# print (len(layers))
