import matplotlib as mpl
from matplotlib import pyplot as plt
import time
import numpy as np
import requests
from io import BytesIO
from PIL import Image

response = requests.get("https://img1.southernliving.timeinc.net/sites/default/files/styles/4_3_horizontal_inbody_900x506/public/image/2018/07/main/the_sunflower_fields_at_neuse_river_greenway_trail.jpg?itok=ZOlvAuIg&1532035249")
img = Image.open(BytesIO(response.content))
img = img.convert('L')
img.save("blob_img.png")

def laplacian_of_gaussian(sigma, size):
    pass
    
size = 19
sigma = 1
a = np.zeros((size,size))
for i in range(size):
    for j in range(size):
        a[j,i] = -(1 - ((19 - j)**2 + (19 - i)**2) / (2 * sigma**2)) * np.exp
