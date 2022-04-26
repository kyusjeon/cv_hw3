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
img.save("corner_img.png")

img = mpl.pyplot.imread("corner_img.png")

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
    
    return  filtered

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

filtered_x = filter(filter_sobelx, img)
filtered_y = filter(filter_sobely, img)

window = np.ones((win_size_list[-1], win_size_list[-1]))

hariss_a = filter(window, filtered_x*filtered_x)
hariss_b = filter(window, filtered_x*filtered_y)
hariss_c = filter(window, filtered_y*filtered_y)