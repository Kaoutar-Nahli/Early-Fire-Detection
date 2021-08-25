#%%
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
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
# %%


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
test_data_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)
# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device
# %%
x = train_data[0][0]
print(x.shape)
#plt.imshow(x.squeeze().numpy(), cmap='gray')
#plt.show()
