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

def gkern(l,):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    sig = int((l-1)/2)
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

# window = gkern((win_size_list[-1], win_size_list[-1]))

# hariss_a = filter(window, filtered_x*filtered_x)
# hariss_b = filter(window, filtered_x*filtered_y)
# hariss_c = filter(window, filtered_y*filtered_y)

def harris_M(a,b,c):
    met = np.array([[a, b],
          [b, c]]).transpose(2,3,0,1)

    return met

def harris_oper(a,b,c):
    oper = np.nan_to_num((a * c - b * b)/(a + c))

    return oper

def cornerDetect(img, winSize=7, type=0):
    start_time = time.time()
    filtered_x = filter(filter_sobelx, img)
    filtered_y = filter(filter_sobely, img)

    window = gkern(winSize)

    harris_a = filter(window, filtered_x*filtered_x)
    harris_b = filter(window, filtered_x*filtered_y)
    harris_c = filter(window, filtered_y*filtered_y)

    # Eigenvalue
    if type == 0:
        harris_m = harris_M(harris_a, harris_b, harris_c)
        eig_val, eig_vec = np.linalg.eig(harris_m)

        return eig_val.min(axis=2)

    # Harris
    elif type == 1:
        harris_h = harris_oper(harris_a, harris_b, harris_c)

        return harris_h
    
time_list_eig, time_list_h = list(), list()

for win_size in win_size_list :
    start_time = time.time()
    dection = cornerDetect(img, win_size, type=0)
    end_time = time.time()
    time_list_eig.append(end_time - start_time)

    print('Window Size = ', win_size)
    print('Time = ', time_list_eig[-1])
    plt.figure(None, figsize=(16,9))
    plt.imshow(dection, cmap='gray')
    plt.show()
    
dection = cornerDetect(img, win_size_list[0], type=0)

response = requests.get("https://miro.medium.com/max/700/1*KvDEGIpfwdJtUFwB5acYEA.png")
cimg = Image.open(BytesIO(response.content))
cimg = cimg.convert('RGB')
cimg.save("corner_cimg.png")

cimg = mpl.pyplot.imread("corner_cimg.png")


a = gkern(21)

x = np.linspace (0, 50, 1000)
y = 0.75 * np.sin(x)

peaks = np.argwhere((y[1:-1] > y[:-2]) * (y[1:-1] > y[2:]))
dips = np.where((y[1:-1] < y[:-2]) * (y[1:-1] < y[2:]))[0] + 1

plt.plot (x, y)
plt.plot (x[peaks], y[peaks], 'o')
plt.plot (x[dips], y[dips], 'o')

plt.show()

y = dection
y = np.where(y>y.max()*0.3, y, y.max()*0.3)
peaks = np.where((y[1:-1,1:-1] > y[:-2,1:-1]) * (y[1:-1,1:-1] > y[2:,1:-1]) * (y[1:-1, 1:-1] > y[1:-1, :-2]) * (y[1:-1,1:-1] > y[1:-1,2:]),1,0)
plt.figure(dpi=500)
plt.imshow(peaks, cmap='gray')
plt.savefig('output.png')