#%%
import torch
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
from sklearn import preprocessing
from sklearn.metrics import f1_score
import torch.optim as optim
drive = False
model_name = 'CNN'

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

class CNN_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3,6,5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,3),
            torch.nn.Conv2d(6,5,5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,3),
            torch.nn.Conv2d(5,5,5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3,3),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(180,500),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(500,120),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(120,20),
            torch.nn.ReLU(),
            torch.nn.Linear(20,3),
            torch.nn.LogSoftmax(dim=1)
        )
    def forward(self, X):
        return self.layers(X)

model = CNN_model()
model.to(device)
#%%
from torch.utils.tensorboard import SummaryWriter
#pip install tensorflow
%load_ext tensorboard
#%%
epochs = 20
lr = 0.0001
lr_str = '0_0001'
optimizer_name ='Adam'
loss_name = 'NLLLoss'
momentum = 0.9

# Define a loss function and optimizer
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum) # create optimiser
criterion = torch.nn.CrossEntropyLoss()



steps = 0
running_loss = 0
print_every = 5
train_losses, test_losses = [], []
run_name = f'{model_name}_{epochs}_{lr_str}_{optimizer_name}_{loss_name} '
if drive:
   writer = SummaryWriter(f'./drive/MyDrive/runs/{run_name}')
else:
  writer = SummaryWriter(f'runs/{run_name}')



def train(model, train_dataloader, epochs=epochs):
    train_losses = []
    test_losses = []
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs): #for each epoch data is scanned in 40 batches of 64 points
        for inputs,labels in train_dataloader:
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            y_hat = model(inputs)
            loss =criterion(y_hat,labels)  # this only works if the target is dim-1 - should use n_labels
            loss.backward()  # Upgrades the .grad -- of each of the parameters (based on backpopulating through the NN)
            optimizer.step()
            # print stat
            running_loss += loss.item()
            writer.add_scalar('loss/train',loss.item(), steps)
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                f1_score_sum = 0
                model.eval() # evaluation mode
                with torch.no_grad():
                    for inputs, labels in test_data_loader:
                        #inputs, labels = inputs.to(device),labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        test_loss += batch_loss.item()
                        ps = torch.exp(logps) 
                        #Returns a new tensor with the exponential of the elements of the input tensor
                        top_p, top_class = ps.topk(1, dim=1)
                        #Returns the k largest elements of the given input tensor along a given dimension
                        equals = top_class == labels.view(*top_class.shape)
                        #Returns a new tensor with the same data as the self tensor but of a different shape
                        accuracy +=torch.mean(equals.type(torch.FloatTensor)).item()
                        f1_score_conv = f1_score(top_class.cpu().detach().numpy(), labels.cpu().detach().numpy(),average='micro')
                        f1_score_sum += f1_score_conv
                train_losses.append(running_loss/print_every)# 
                test_losses.append(test_loss/len(test_data_loader))               
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. " 
                    f"Test loss: {test_loss/len(test_data_loader):.3f}.. "
                    f"Test accuracy: {accuracy/len(test_data_loader):.3f}")
                writer.add_scalar('loss/test',test_loss/len(test_data_loader), steps)
                writer.add_scalar('accuracy/test',accuracy/len(test_data_loader), steps)
                writer.add_scalar('F1_score/test',f1_score_sum/len(test_data_loader), steps)
                running_loss = 0
    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.show()
   # plt.savefig('./drive/MyDrive/models/CNN_Model_Loss_1_epochs.png')

train(model, train_data_loader, epochs=epochs)


current_time = str(datetime.now().timestamp())
if drive:
    path_models = f'./drive/MyDrive/models/{model_name}_{epochs}_{lr_str}_{optimizer_name}_{loss_name}_{current_time}.pt'
else:
    path_models = f'tests/{model_name}_{epochs}_{lr_str}_{optimizer_name}_{loss_name}_{current_time}.pt'


torch.save(model.state_dict(), path_models)
# main errors were related to flatten and convert y_hat to top_class
# %%

#%%
#loading back model
# net = CNN_model()
# net.load_state_dict(torch.load('./drive/MyDrive/models/CNN_01.pt.'))

#%%
