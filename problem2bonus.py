from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

img = mpl.pyplot.imread("blob_img.png")

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

filtered_x = filter(filter_sobelx/8, img)
filtered_y = filter(filter_sobely/8, img)

def sobel_temperature(img):
    filtered_x = filter(filter_sobelx/8, img)
    filtered_xx = filter(filter_sobelx/8, filtered_x)
    filtered_y = filter(filter_sobely/8, img)
    filtered_yy = filter(filter_sobely/8, filtered_y)
    temperature = img + filtered_xx + filtered_yy
    temperature /= temperature.max()
    
    return temperature

def cal_temperature(img):
    h, w = img.shape
    for _i in np.arange(1,h - 1,1):
        for _j in np.arange(1,w - 1,1):
            img[_i, _j] += (img[_i - 1, _j] - 2 * img[_i, _j] + img[_i + 1, _j]) / 2**2 + (img[_i, _j - 1] - 2 * img[_i, _j] + img[_i, _j + 1]) / 2**2

    return img / img.max()

for _i in range(100):
    img = cal_temperature(img)
    print(_i)
    if _i % 10 == 0:
        plt.imshow(img, cmap='gray')
        plt.show()
