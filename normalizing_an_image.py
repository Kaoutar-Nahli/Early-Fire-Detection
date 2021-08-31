#%%
# calculate the mean and standard deviation of an image and normalize it

# import necessary libraries
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
  
# load the image
img_path = "data\FIRE-SMOKE-DATASET\Test\Fire\image_1.jpg"
img = Image.open(img_path)
  
# convert PIL image to numpy array
img_np = np.array(img)
  
# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels") 


#%%
# Python code for converting PIL Image to
# PyTorch Tensor image and plot pixel values
  
# import necessary libraries
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
  
# define custom transform function
transform = transforms.Compose([
    transforms.ToTensor()
])
  
# transform the pIL image to tensor image
img_tr = transform(img)
  
# Convert tensor image to numpy array
img_np = np.array(img_tr)
  
# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")

#%%
# Python code to calculate mean and std of image
  
# get tensor image
img_tr = transform(img)
  
# calculate mean and std
mean, std = img_tr.mean([1,2]), img_tr.std([1,2])
  
# print mean and std
print("mean and std before normalize:")
print("Mean of the image:", mean)
print("Std of the image:", std)


# python code to normalize the image
  
#%% 
from torchvision import transforms
  
# define custom transform
# here we are using our calculated
# mean & std
transform_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
  
# get normalized image
img_normalized = transform_norm(img)
  
# convert normalized image to numpy
# array
img_np = np.array(img_normalized)
  
# plot the pixel values
plt.hist(img_np.ravel(), bins=50, density=True)
plt.xlabel("pixel values")
plt.ylabel("relative frequency")
plt.title("distribution of pixels")

# %%
