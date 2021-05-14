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
# for reading files from urls
import urllib.request
# display imports
from IPython.display import display, IFrame
from IPython.core.display import HTML
import logging
import json, logging.config

class ProcessImage(object):

    segList = None

    # def show_images(images: List[numpy.ndarray]) -> None:
    def show_images(self, images):
        n: int = len(images)
        f = plt.figure(figsize=(16,20), dpi=80)
        for i in range(n):
            # Debug, plot figure
            # f.add_subplot(1, n, i + 1)
            ax = f.add_subplot(4, 2, i + 1)
            ax.title.set_text(images[i][0])
            plt.imshow(images[i][1], cmap="gray")

        plt.show(block=True)

    def plot5(self, seg):
        keys = list(seg.images.keys())
        n: int = len(keys)
        f = plt.figure(figsize=(10,10))
        for i in range(n):
            # Debug, plot figure
            # f.add_subplot(1, n, i + 1)
            ax = f.add_subplot(2, 3, i + 1)
            ax.title.set_text(keys[i])
            plt.imshow(seg.images.get(keys[i]).get('image'))
        plt.show(block=True)

    def plot4(self, seg):
        list = []
        list.append(("original", seg.image))
        list.append(("GT", seg.imageGT))
        list.append(("Box ROI", seg.imageBoxROI))
        list.append(("internal zone", seg.imageROICropped))
        list.append(("posterior zone", seg.imagePosterior))
        list.append(("marginal zone", seg.imageMarginal))
        list.append(("Canny Edge", seg.imageCannyEdge))
        list.append(("Contour", seg.imageContours))
        output = {"images":list}
        self.show_images(list)

    def plot(self, seg):
        fig = plt.figure()
        ax = fig.add_subplot(131, projection='3d')
        ax.imshow(seg.image)
        plt.show()
        pass

    def plot2(self, seg):
        Hori = np.concatenate((seg.image, seg.imageGT, seg.imageCorner), axis=1)
        # cv2.imshow('Region Growing', seg.image)
        # cv2.imshow('Region Growing', seg.imageGT)
        cv2.imshow('Horizontal', Hori)
        cv2.waitKey()
        cv2.destroyAllWindows()

    def plot3(self, seg):
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

    def display6(self, ids):
        # ids = (294,)
        # ids = np.arange(1,148)
        segList = BUSSegmentorList()
        segList.loadDataSetB(ids)
        size = len(segList.BUSList)
        seg = segList.BUSList[size -1]
        seg.createCornerImage()
        seg.cropBoxROI()
        seg.cropContourROI()
        seg.cropPosteriorZone()
        seg.cropMarginalZone()
        seg.createCannyEdgedImage()
        self.plot4(seg)

    def display7(self, ids):
        self.load(ids)
        size = len(self.segList.BUSList)
        seg = self.segList.BUSList[size -1]
        seg.findContours(addImages=True)
        self.plot5(seg)

    def runSaveGTStats(self, ids):
        segList = BUSSegmentorList()
        segList.loadDataSetB(ids)
        size = len(segList.BUSList)
        print(f"segList size={size}")
        segList.saveGTStats()

    def load(self, ids):
        self.segList = BUSSegmentorList()
        self.segList.loadDataSetB(ids)
        size = len(self.segList.BUSList)
        print(f"segList size={size}")


    def main(self):
        import os
        import sys
        localPath = "c:\\Users\\djhalama\\Documents\\GitHub\\BUS"
        if os.path.exists(localPath):
            # print(f"localPath exists: {localPath}")
            os.chdir("c:\\Users\\djhalama\\Documents\\GitHub\\BUS")
        else:
            from google.colab import drive
            drive.mount('/content/gdrive')
            os.chdir('/content/gdrive/MyDrive/BUS_Project_Home/Share_with_group/David_Halama/BUS')
            sys.path.append('/content/gdrive/MyDrive/BUS_Project_Home/Share_with_group/David_Halama/BUS/Segmentation')
        from Common import Common
        # https://coralogix.com/log-analytics-blog/python-logging-best-practices-tips/
        with open('logging-config.json', 'rt') as f:
            config = json.load(f)
            logging.config.dictConfig(config)





