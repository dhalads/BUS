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
import cv2
import pandas as pd
from BUSSegmentor import BUSSegmentor
import logging


class BUSSegmentorList(object):

    logger = logging.getLogger("BUS." + __name__)
    BUSList = []

    def __init__(self):
        pass

    def loadDataSetB(self, ids):
        try:
            pass
            for id in ids:
                seg = BUSSegmentor()
                name = str(id).zfill(6) +".png"
                seg.loadImage(name)
                seg.loadImageGT()
                self.BUSList.append(seg)

        except Exception as e:
            pass
            # self.logger.error(e)
            # self.logger.exception(e)
            self.logger.exception("help me")
            raise
        else:
            pass
        finally:
            pass