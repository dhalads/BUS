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
    isError = False
    errorMessages =  []

    def __init__(self):
        pass

    def loadDataSetB(self, ids):
        countFileNotFound = 0
        try:
            for id in ids:
                try:
                    seg = BUSSegmentor()
                    seg.id = id
                    name = str(id).zfill(6) +".png"
                    seg.loadImage(name)
                    seg.loadImageGT()
                    self.BUSList.append(seg)
                    # raise Exception("Made it ok.")
                except FileNotFoundError as nf:
                    self.isError = True
                    msg = f"file not found name={name}, id={id}"
                    self.logger.error(msg)
                    # self.logger.error('file not found name=%s, id=%s', name, id)
                    self.errorMessages.append(msg)
                    countFileNotFound += 1
                    pass
                except Exception as e:
                    pass
                    # self.logger.error(e)
                    # self.logger.exception(e)
                    self.logger.exception("help me")
                    raise
        except Exception as e:
            pass
            # self.logger.error(e)
            # self.logger.exception(e)
            self.logger.exception("help me2")
            raise
        else:
            pass
        finally:
            pass
        self.logger.debug("Results loadDataSetB: loaded=%s, failed=%s", len(self.BUSList), countFileNotFound, exc_info=True)

    def saveGTStats(self):
        output = []
        try:
            pass
            for seg in self.BUSList:
                cnt = seg.getGTContour()
                if cnt is not None :
                    stats = seg.createContourStats(None, cnt)
                    output.append(stats)
                else:
                    self.logger.debug('Did not have GT contour name=%s', seg.imageName)
            df = pd.DataFrame(output)
            df.to_csv('statsGT.csv')
        except Exception as e:
            pass
            # self.logger.error(e)
            self.logger.exception(e)
            raise

    def saveROIStats(self):
        output = []
        try:
            pass
            for seg in self.BUSList:
                cnt, stats = seg.findContours()
                if cnt is not None :
                    output.append(stats)
                else:
                    self.logger.debug('Did not find contour name=%s', seg.imageName)
                    stats = {}
                    stats['id'] = seg.id
                    stats['imageName'] = seg.imageName
                    output.append(stats)
            df = pd.DataFrame(output)
            df[df.columns.difference(["cnt"])].to_csv('statsROI.csv')
        except Exception as e:
            pass
            # self.logger.error(e)
            self.logger.exception(e)
            raise
