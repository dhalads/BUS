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
import os.path
from pathlib import Path
from Common import Common

class BUSSegmentor(object):

    image = None
    imageName = None
    imageGT = None
    imageCorner = None

    def __init__(self):
        pass

    def loadImage(self, filename):
        self.imageName = filename
        path = Common.getImagePath()
        path = path / "original" / filename
        image = Image.open(path).convert('L') # Make sure to convert to grayscale
        image_inv = ImageOps.invert(image)
        bus = asarray(image_inv)
        self.image = bus

    def loadImageGT(self):
        path = Common.getImagePath()
        path = path / "GT" / self.imageName
        image = Image.open(path).convert('L') # Make sure to convert to grayscale
        image_inv = ImageOps.invert(image)
        bus = asarray(image_inv)
        self.imageGT = bus

    def examineFilename(self, filename):
        # pathfile = Path(filename)
        # dir = pathfile.dirname()
        # name = pathfile.filename()
        path = Common.getImagePath()
        path = path / "original" / filename
        return path

    def createCornerImage(self):
        self.imageCorner = self.image.copy()
        x = self.image.shape[0]
        y = self.image.shape[1]
        for i in range(x%10):
            for j in range(y%10):
                self.imageCorner[x, y] = 255
        pass