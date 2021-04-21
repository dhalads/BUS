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


# def show_images(images: List[numpy.ndarray]) -> None:
def show_images(images):
    n: int = len(images)
    f = plt.figure(figsize=(16,20))
    for i in range(n):
        # Debug, plot figure
        # f.add_subplot(1, n, i + 1)
        ax = f.add_subplot(3, 2, i + 1)
        ax.title.set_text(images[i][0])
        plt.imshow(images[i][1])

    plt.show(block=True)

def plot4():
    list = []
    list.append(("original", seg.image))
    list.append(("GT", seg.imageGT))
    list.append(("Box ROI", seg.imageBoxROI))
    list.append(("internal zone", seg.imageROICropped))
    list.append(("posterior zone", seg.imagePosterior))
    list.append(("marginal zone", seg.imageMarginal))
    output = {"images":list}
    show_images(list)

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
    cv2.imshow('ROI', seg.imageGT)
    cv2.imshow('corner', seg.imageCorner)
    cv2.waitKey()
    cv2.destroyAllWindows()

seg = BUSSegmentor()
seg.loadImage("000002.png")
print(seg.image.shape)
seg.loadImageGT()
print(seg.imageGT.shape)
seg.createCornerImage()
print(seg.image[0][0])
print(seg.imageGT[0][0])
print(seg.imageCorner[0][0])
print(type(seg.image))
print(type(seg.imageGT))
print(type(seg.imageCorner))
seg.cropBoxROI()
seg.cropContourROI()
seg.cropPosteriorZone()
seg.cropMarginalZone()
plot4()



