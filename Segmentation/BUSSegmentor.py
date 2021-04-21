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

class BUSSegmentor(object):

    image = None
    imageName = None
    imageGT = None
    imageCorner = None
    imageBoxROI = None
    imageROICropped = None
    imagePosterior = None
    imageMarginal = None

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
        for i in range(x-1,x-30, -1):
            for j in range(30):
                self.imageCorner[i][j] = 255
        pass

    def cropBoxROI(self):
        edges= cv2.Canny(self.imageGT, 50,200)
        contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        for (i,c) in enumerate(sorted_contours):
            x,y,w,h= cv2.boundingRect(c)
            cropped_contour= self.image[y:y+h, x:x+w]
        self.imageBoxROI = cropped_contour

    def cropContourROI(self):
        # https://stackoverflow.com/questions/28759253/how-to-crop-the-internal-area-of-a-contour
        edges= cv2.Canny(self.imageGT, 50,200)
        contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        idx = 0 # The index of the contour that surrounds your object
        mask = np.zeros_like(self.image) # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
        out = np.zeros_like(self.image) # Extract out the object and place into output image
        out[mask == 255] = self.image[mask == 255]
        self.imageROICropped = out

    def cropPosteriorZone(self):
        # https://www.pyimagesearch.com/2015/02/09/removing-contours-image-using-python-opencv/
        edges= cv2.Canny(self.imageGT, 50,200)
        contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for (i,c) in enumerate(contours):
            x,y,w,h= cv2.boundingRect(c)
            mask = np.ones(self.image.shape[:2], dtype="uint8") * 255
            cv2.drawContours(mask, [c], -1, 0, -1)
        limage = cv2.bitwise_and(self.image, self.image, mask=mask)
        imageh, imagew = self.image.shape
        self.imagePosterior= limage[y+h//2:imageh, x:x+w]

    def cropMarginalZone(self):
        edges= cv2.Canny(self.imageGT, 50,200)
        contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for (i,c) in enumerate(contours):
            contours[i] = self.scale_contour(c, 1.5)
            mask_in = np.ones(self.image.shape[:2], dtype="uint8") * 255

        idx = 0 # The index of the contour that surrounds your object
        mask_out = np.zeros_like(self.image) # Create mask where white is what we want, black otherwise
        cv2.drawContours(mask_out, contours, idx, 255, -1) # Draw filled contour in mask
        out = np.zeros_like(self.image) # Extract out the object and place into output image
        out[mask_out == 255] = self.image[mask_out == 255]
        cv2.drawContours(mask_in, [c], -1, 0, -1)
        limage = cv2.bitwise_and(out, out, mask=mask_in)
        self.imageMarginal = limage

    def cropMarginalZone2(self):
        edges= cv2.Canny(self.imageGT, 50,200)
        contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        for (i,c) in enumerate(sorted_contours):
            c = self.scale_contour(c, 1.5)
            x,y,w,h= cv2.boundingRect(c)
            cropped_contour= self.image[y:y+h, x:x+w]
        self.imageMarginal = cropped_contour

    def scale_contour(self, cnt, scale):
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)

        return cnt_scaled