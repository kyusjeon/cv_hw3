from cv2 import threshold
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

img = mpl.pyplot.imread("blob_img.png")

def laplacian_of_gaussian(sigma, size):
    margin = int(size / 2)
    filter = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            filter[j,i] = -(1 - ((margin - j)**2 + (margin - i)**2) / (2 * sigma**2)) * np.exp(-((margin - j)**2 + (margin - i)**2) / (2 * sigma**2)) / (np.pi * sigma**4)
            
    return filter

def filter_img(img, fil):
    # these two lines are important as they ensure correct
    # computations!
    fil = fil.astype(float)    # convert correctly
    img = img.astype(float)            # convert correctly
    # this holds the end result
    filtered = np.zeros_like(img)
    width = int((fil.shape[1]-1)/2)
    height = int((fil.shape[0]-1)/2)
    # do the filtering
    for i in range(height,img.shape[1]-height):
        for j in range(width,img.shape[0]-width):
            filtered[j,i] = (fil * img[j-width:j+width+1, i-height:i+height+1]).sum()       # how do you create the output of the filtering?
    
    return  filtered

def track_scale(img, sigma, size, threshold):
    log = laplacian_of_gaussian(sigma, size)
    blob_img = filter_img(img, log)
    v_max = blob_img.max()
    mask = blob_img > v_max * threshold
    
    return mask, v_max

sigma_list = np.arange(1,7,1)
fil_max_scale = list()
size = 19
thr = 0.5

fig, ax = plt.subplots(1, dpi=400)
ax.imshow(img, cmap='gray')
ax.axis('off')
for sigma in sigma_list:
    mask, v_max = track_scale(img, sigma, size, thr)
    fil_max_scale.append(v_max)
    for y,x in np.argwhere(mask == True):
        circ = mpl.patches.Circle((x, y), sigma, color='red', fill=False)
        ax.add_patch(circ)
plt.show()

plt.figure()
plt.plot()
plt.show()

