# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# import os
# import sys
# localPath = "c:\\Users\\djhalama\\Documents\\GitHub\\BUS"
# if os.path.exists(localPath):
#     # print(f"localPath exists: {localPath}")
#     os.chdir("c:\\Users\\djhalama\\Documents\\GitHub\\BUS")
# else:
#     from google.colab import drive
#     drive.mount('/content/gdrive')
#     os.chdir('/content/gdrive/MyDrive/BUS_Project_Home/Share_with_group/David_Halama/BUS')
#     sys.path.append('/content/gdrive/MyDrive/BUS_Project_Home/Share_with_group/David_Halama/BUS/Segmentation')
#     from Common import Common
#     # print(Common.getImagePath())

# print(sys.path)
# print(os.getcwd())

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

from ProcessImage import ProcessImage
pimg = ProcessImage()
pimg.main()

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.stats import multivariate_normal
import numpy as np
import scipy.misc
import scipy.ndimage
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage import segmentation
from skimage.filters import sobel
from skimage.color import label2rgb
from numpy import asarray
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import cv2
from BUSSegmentor import BUSSegmentor
from BUSSegmentorList import BUSSegmentorList
from ProcessImage import ProcessImage
from ProcessImage import busUI
# for reading files from urls
import urllib.request
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML
import logging
import json, logging.config
import importlib


# %run -i ProcessImage.py
# display(seg.contourStats)

# importlib.reload(BUSSegmentor)

def run():
    # pimg.load(np.arange(1,144)) #80, 101, 125
    # pimg.display7((80,))
    # pimg.runSaveGTStats(np.arange(1, 144))
    # pimg.segList.saveROIStats()
    # pimg.useBoxes()
    bus = busUI()
    bus.initUI()


run()


# %%



# %%



# %%



