from matplotlib import pyplot as plt
from io import BytesIO
from PIL import Image
import matplotlib as mpl
import numpy as np
import requests
import time


response = requests.get("https://img.freepik.com/free-vector/abstract-distorted-checkered-background_23-2148965764.jpg?t=st=1650245746~exp=1650246346~hmac=29a27d58604684adf6c7a8140834106058f08462598935fa2273b65e54f72461&w=1800")
img = Image.open(BytesIO(response.content))
img = img.convert('L')
img.save("test_img.png")

img = mpl.pyplot.imread("test_img.png")

win_size_list = [3, 5, 7, 9, 11, 15, 21]

method_type = {0:"Eigenvalue", 1:"Harris"}

def filter(weights,roi):
    # these two lines are important as they ensure correct
    # computations!
    weights = weights.astype(float)    # convert correctly
    roi = roi.astype(float)            # convert correctly
    # this holds the end result
    filtered = np.zeros_like(roi)
    width = int((weights.shape[1]-1)/2)
    height = int((weights.shape[0]-1)/2)
    # do the filtering
    for i in range(height,roi.shape[1]-height):
        for j in range(width,roi.shape[0]-width):
            filtered[j,i] = (weights * roi[j-width:j+width+1, i-height:i+height+1]).sum()       # how do you create the output of the filtering?
    
    return  filtered.sum()

filter_sobelx = np.array([
[-1,0,1],
[-2,0,2],
[-1,0,1],
])

filter_sobely = np.array([
[1,2,1],
[0,0,0],
[-1,-2,-1],
])

def intensity(weights,img, window_size):
    # these two lines are important as they ensure correct
    # computations!
    weights = weights.astype(float)    # convert correctly
    img = img.astype(float)            # convert correctly
    # this holds the end result
    filtered = np.zeros_like(img)
    width = int((window_size-1)/2)
    height = int((window_size-1)/2)
    # do the filtering
    for i in range(height,img.shape[1]-height):
        for j in range(width,img.shape[0]-width):
            filtered[j,i] = filter(weights, img[j-width:j+width+1, i-height:i+height+1])
    
    return  filtered

filtered_x = intensity(filter_sobelx, img, win_size_list[-1])
filtered_y = intensity(filter_sobely, img, win_size_list[-1])

