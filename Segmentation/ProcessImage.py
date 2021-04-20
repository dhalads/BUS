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

def plot():
    fig = plt.figure()
    ax = fig.add_subplot(131, projection='3d')
    ax.imshow(seg.image)
    plt.show()
    pass

def plot2():
    Hori = np.concatenate((seg.image, seg.imageGT, seg.imageCorner), axis=1)
    # cv2.imshow('Region Growing', seg.image)
    # cv2.imshow('Region Growing', seg.imageGT)
    cv2.imshow('Horizontal', Hori)
    cv2.waitKey()
    cv2.destroyAllWindows()

def plot3():
    Hori = np.concatenate((seg.image, seg.imageGT, seg.imageCorner), axis=1)
    # cv2.imshow('Region Growing', seg.image)
    # cv2.imshow('Region Growing', seg.imageGT)
    cv2.namedWindow("original")
    cv2.namedWindow("GT")
    cv2.namedWindow("corner")
    cv2.imshow('original', seg.image)
    cv2.imshow('GT', seg.imageGT)
    cv2.imshow('corner', seg.imageCorner)
    cv2.waitKey()
    cv2.destroyAllWindows()

seg = BUSSegmentor()
seg.loadImage("000002.png")
print(seg.image.shape)
print(type(seg.image))
seg.loadImageGT()
print(seg.imageGT.shape)
seg.createCornerImage()
plot2()



