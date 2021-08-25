#%%
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


#%%
def load_split_train_test(data_directory, valid_size = 0.2):

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
    val_data = datasets.ImageFolder(root='data\FIRE-SMOKE-DATASET\Train', transform=transforms_test)
    test_data = datasets.ImageFolder(root='data\FIRE-SMOKE-DATASET\Test', transform=transforms_test)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_data_loader = torch.utils.data.DataLoader(train_data,sampler=train_sampler, batch_size=64, shuffle=True)
    val_data_loader = torch.utils.data.DataLoader(train_data,sampler=val_sampler, batch_size=64, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
    return train_data_loader, val_data_loader, test_data_loader




#%%
#spliting the data
#train_data, val_data = torch.utils.data.random_split(train_data, [2400,300])
#%%
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
#add SubsetRandomSampler:
#  https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5





#%%
x = train_data[0][0]
print(x.shape)
#plt.imshow(x.squeeze().numpy(), cmap='gray')
#plt.show()

#%%
with Image.open("data\FIRE-SMOKE-DATASET\Test\Fire\image_1.jpg") as im:
    im.rotate(90).show()
# %%
#create thumnails
# from PIL import Image
# import glob, os

# size = 128, 128

# for infile in glob.glob("data\FIRE-SMOKE-DATASET\Test\Fire\*.jpg"):
#     file, ext = os.path.splitext(infile)
#     with Image.open(infile) as im:
#         im.thumbnail(size)
#         im.save(file + ".thumbnail", "JPEG")