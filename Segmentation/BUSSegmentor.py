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
import logging


class BUSSegmentor(object):

    logger = logging.getLogger("BUS." + __name__)

    images = {}
    image = None
    imageName = None
    imageGT = None
    imageCorner = None
    imageBoxROI = None
    imageROICropped = None
    imagePosterior = None
    imageMarginal = None
    imageCannyEdge = None
    imageContours = None
    contourStats = None

    def __init__(self):
        pass
        self.logger.debug("in init")

    def addImage(self, name, image, roiStats, annotation, response):
        output = {}
        output['image'] = image
        output['roiStats'] = roiStats
        output['annotations']= annotation
        output['response'] = response
        self.images[name] = output

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
        # https://medium.com/analytics-vidhya/tutorial-how-to-scale-and-rotate-contours-in-opencv-using-python-f48be59c35a2
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        cnt_norm = cnt - [cx, cy]
        cnt_scaled = cnt_norm * scale
        cnt_scaled = cnt_scaled + [cx, cy]
        cnt_scaled = cnt_scaled.astype(np.int32)

        return cnt_scaled

    def createCannyEdgedImage(self):
        print("help2")
        # gray=cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        edged=cv2.Canny(self.image,30,200)
        self.imageCannyEdge = edged
        contourImage = self.image.copy()
        # contourImage = cv2.blur(contourImage, (5,5))
        mean = cv2.mean(contourImage)
        mean = int(mean[0])
        mythres = 255 - (255-mean)*0.5
        ret, thresh = cv2.threshold(contourImage, mythres, 255, type=0)
        contours, hierarchy= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.imageCannyEdge = thresh
        # self.imageContours = contourImage
        # print(contours)
        print('Numbers of contours found=' + str(len(contours)))
        stats = [self.createContourStats(x, contours[x]) for x in range(len(contours))]
        # print(stats)
        df = pd.DataFrame(stats)
        self.contourStats = df
        df[df.columns.difference(["cnt"])].to_csv('stats.csv')
        filter_cnt = df.loc[(df['area'] > 200) & (df['area'] < 8000000)]
        # sort_cnt = df.sort_values(by=['area'], ascending=False, inplace=True).head(n=10)
        tmpImg = self.image.copy()
        cntList = filter_cnt['cnt'].tolist()
        print('Numbers of contours plotted=' + str(len(cntList)))
        cv2.drawContours(tmpImg, cntList, -1, (0,255,0))
        self.imageContours = tmpImg
        # stats = regionprops(BW, 'basic')

    def createContourStats(self, id, cnt):
        output = {}
        output["id"] = id
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        area = cv2.contourArea(cnt)
        output['aspect_ratio'] = aspect_ratio
        output['area'] = area

        # x,y,w,h = cv.boundingRect(cnt)
        rect_area = w*h
        extent = float(area)/rect_area
        output['extent'] = extent

        # area = cv.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area != 0 :
            solidity = float(area)/hull_area
        else:
            solidity = 0
        output['solidity'] = solidity

        # area = cv.contourArea(cnt)
        equi_diameter = np.sqrt(4*area/np.pi)

        if cnt.shape[0] >= 5 :
            (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
        else:
            angle = None
        output['angle'] = angle

        mask = np.zeros(self.image.shape,np.uint8)
        cv2.drawContours(mask,[cnt],0,255,-1)
        pixelpoints = np.transpose(np.nonzero(mask))
        #pixelpoints = cv.findNonZero(mask)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.image,mask = mask)
        output['min_val'] = min_val
        output['max_val'] = max_val
        output['min_loc'] = min_loc
        output['max_loc'] = max_loc


        mean_val = cv2.mean(self.image,mask = mask)
        output['mean_val'] = mean_val

        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        output['leftmost'] = leftmost
        output['rightmost'] = rightmost
        output['topmost'] = topmost
        output['bottommost'] = bottommost

        # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
        M = cv2.moments(cnt)
        if area > 0 :
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
        else:
            cx = None
            cy = None
        output['cx'] = cx
        output['cy'] = cy
        perimeter = cv2.arcLength(cnt,True)
        output['perimeter'] = perimeter
        output['cnt'] = cnt

        return(output)

    def combineContours(self):
        # https://stackoverflow.com/questions/44501723/how-to-merge-contours-in-opencv
        list_of_pts = []
        # for ctr in ctrs_to_merge
        #     list_of_pts += [pt[0] for pt in ctr]
        # ctr = np.array(list_of_pts).reshape((-1,1,2)).astype(np.int32)
        # ctr = cv2.convexHull(ctr) # done.
        # https://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them

