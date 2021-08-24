#%%
from PIL import Image
with Image.open("data\FIRE-SMOKE-DATASET\Test\Fire\image_1.jpg") as im:
    im.rotate(90).show()
# %%
