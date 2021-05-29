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
from exceptions import SegmentationError
import cv2
import pandas as pd
import logging


class BUSSegmentor(object):

    logger = logging.getLogger("BUS." + __name__)

    def __init__(self):
        pass
        self.logger.debug("in init")
        self.id = None
        self.images = {}
        self.image = None
        self.PILimage = None
        self.imageName = None
        self.imageGT = None
        self.PILimageGT = None
        self.imageCorner = None
        self.imageBoxROI = None
        self.imageROICropped = None
        self.imagePosterior = None
        self.imageMarginal = None
        self.imageCannyEdge = None
        self.imageContours = None
        self.contourStats = None

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
        try:
            image = Image.open(path).convert('L') # Make sure to convert to grayscale
        except FileNotFoundError:
            filename = str(self.id) + ".png"
            path = Common.getImagePath() / "original" / filename
            image = Image.open(path).convert('L')
        self.PILimage= image
        # image_inv = ImageOps.invert(image)
        bus = asarray(image)
        self.image = bus

    def loadImageGT(self):
        path = Common.getImagePath()
        path = path / "GT" / self.imageName
        image = Image.open(path).convert('L') # Make sure to convert to grayscale
        self.PILimageGT = image
        # image_inv = ImageOps.invert(image)
        bus = asarray(image)
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

    def getGTContour(self):
        # contourImage = self.image.copy()
        # # contourImage = cv2.blur(contourImage, (5,5))
        # mean = cv2.mean(contourImage)
        # mean = int(mean[0])
        # mythres = 255 - (255-mean)*0.5
        # ret, thresh = cv2.threshold(contourImage, mythres, 255, type=0)
        # contours, hierarchy= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours, hierarchy= cv2.findContours(self.imageGT, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours= sorted(contours, key=cv2.contourArea, reverse= True)
        size = len(sorted_contours)
        if size>0 :
            return(sorted_contours[0])
        else:
            return None
            # raise SegmentationError("failed to get contour")

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

    def findContours(self, addImages=False):
        # https://automaticaddison.com/how-to-detect-and-draw-contours-in-images-using-opencv/
        orig = asarray(self.PILimage)
        if addImages:
            self.addImage("Original", orig, None, None, None)
            self.addImage("GT", self.imageGT, None, None, None)
        tmpImg = self.image.copy()
        GTcnt = self.getGTContour()
        cv2.drawContours(tmpImg, GTcnt , -1, (0,255,0), 5)
        if addImages:
            self.addImage("Original with GT", tmpImg, None, None, None)
        # Convert the grayscale image to binary
        # image_inv = ImageOps.invert(self.PILimage)
        # tmpImg = asarray(image_inv)
        # if addImages:
        #     self.addImage("Inverted Original 1", tmpImg, None, None, None)
        tmpImg = cv2.bitwise_not(self.image.copy())
        if addImages:
            self.addImage("Inverted Original", tmpImg, None, None, None)
        mean = cv2.mean(tmpImg)
        mean = int(mean[0])
        mythres = 255 - (255-mean)*0.5
        ret, binary = cv2.threshold(tmpImg, mythres, 255, 0)
        if addImages:
            self.addImage("Binary", binary, None, None, None)

        # Find the contours on the inverted binary image, and store them in a list
        # Contours are drawn around white blobs.
        # hierarchy variable contains info on the relationship between the contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Draw the contours (in red) on the original image and display the result
        # Input color code is in BGR (blue, green, red) format
        # -1 means to draw all contours
        with_contours = cv2.drawContours(self.image.copy(), contours, -1,(255, 255, 255),1)
        if addImages:
            self.addImage("With Contours", with_contours, None, None, None)

        # Show the total number of contours that were detected
        self.logger.debug('Total number of contours detected: %s', str(len(contours)))
        stats = [self.createContourStats(x, contours[x], GTcnt) for x in range(len(contours))]
        # print(stats)
        df = pd.DataFrame(stats)
        self.contourStats = df
        # df[df.columns.difference(["cnt"])].to_csv('stats.csv')
        df1 = df.loc[(df['area'] > 300) & (df['area'] < 100000)]
        shape1 = self.image.shape
        df2 = df1.loc[(df1['leftx'] > 30 ) & (df1['rightx'] < shape1[1] - 30) & (df1['topy'] > 30) & (df1['bottomy'] < shape1[0] - 30) & (df1['aspect_ratio'] < 5.5)]
        # df2.sort_values(by=['area'], ascending=False, inplace=True)
        df2.sort_values(by=['mean_val'], ascending=True, inplace=True)
        self.logger.debug("\n" + str(df2[df2.columns.difference(["cnt"])]))
        tmpImg = self.image.copy()
        cntList = df2['cnt'].tolist()
        self.logger.debug("Numbers of contours plotted=%s", str(len(cntList)))
        if len(cntList) > 0 :
            bestCntId = df2['cnt_id'].tolist()[0]
            if addImages:
                cv2.drawContours(tmpImg, cntList, 0, (255,255,255), 1)
                self.addImage("WithFilteredContours", tmpImg, None, None, None)
            bestCnt = cntList[0]
            bestStats = stats[bestCntId]
        else:
            bestCnt = None
            bestStats = None
        return((bestCnt, bestStats))

    def createCannyEdgedImage(self):
        print("help2")
        #
        edged=cv2.Canny(self.image,30,200)
        self.imageCannyEdge = edged
        contourImage = self.image.copy()
        # gray=cv2.cvtColor(contourImage,cv2.COLOR_BGR2GRAY)
        # contourImage = cv2.blur(contourImage, (5,5))
        mean = cv2.mean(contourImage)
        mean = int(mean[0])
        mythres = 255 - (255-mean)*0.5
        ret, thresh = cv2.threshold(contourImage, mythres, 255, type=0)
        contours, hierarchy= cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        inverted_binary = ~thresh
        self.imageCannyEdge = thresh
        # self.imageContours = contourImage
        # print(contours)
        print('Numbers of contours found=' + str(len(contours)))
        stats = [self.createContourStats(x, contours[x]) for x in range(len(contours))]
        # print(stats)
        df = pd.DataFrame(stats)
        self.contourStats = df
        # df[df.columns.difference(["cnt"])].to_csv('stats.csv')
        filter_cnt = df.loc[(df['area'] > 200) & (df['area'] < 8000000)]
        # sort_cnt = df.sort_values(by=['area'], ascending=False, inplace=True).head(n=10)
        tmpImg = self.image.copy()
        cntList = filter_cnt['cnt'].tolist()
        print('Numbers of contours plotted=' + str(len(cntList)))
        cv2.drawContours(tmpImg, cntList, -1, (0,255,0), 10)
        self.imageContours = tmpImg
        # stats = regionprops(BW, 'basic')

    def createContourStats(self, cnt_id, cnt, GTcnt=None):
        output = {}
        output['id'] = self.id
        output['imageName'] = self.imageName
        if cnt_id is not None:
            output["cnt_id"] = cnt_id
        x,y,w,h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        area = cv2.contourArea(cnt)
        output['aspect_ratio'] = aspect_ratio
        output['area'] = area

        if GTcnt is not None :
            areaGT = cv2.contourArea(GTcnt)
            areaFracGT = area/areaGT
            output['areaFracGT'] = areaFracGT

        output['leftx'] = x
        output['rightx'] = x + w
        output['topy'] = y
        output['bottomy'] = y + h
        output['width'] = w
        output['height'] = h

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
        output['mean_val'] = mean_val[0]

        leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
        rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
        topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
        bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
        # output['leftmost'] = leftmost
        # output['rightmost'] = rightmost
        # output['topmost'] = topmost
        # output['bottommost'] = bottommost

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
        if (cx is not None) and (GTcnt is not None) :
            # https://stackoverflow.com/questions/50670326/how-to-check-if-point-is-placed-inside-contour
            dist = cv2.pointPolygonTest(GTcnt,(cx,cy),False)
            output['dist'] = dist
        perimeter = cv2.arcLength(cnt,True)
        output['perimeter'] = perimeter
        if cnt_id is not None:
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

