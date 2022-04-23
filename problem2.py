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
img.save("test_img2.png")