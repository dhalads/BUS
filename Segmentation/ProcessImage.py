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
from ipywidgets import widgets, interactive
import pandas as pd
from io import StringIO
import io
import cv2
from ipydatagrid import DataGrid
# from beakerx import *
# from beakerx.object import beakerx


class ProcessImage(object):

    logger = logging.getLogger("BUS." + __name__)

    def __init__(self):
        self.segList = None

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

class busUI(object):

    logger = logging.getLogger("BUS." + __name__)

    def __init__(self):
        self.segList = None

    def load(self, ids):
        self.segList = BUSSegmentorList()
        self.segList.loadDataSetB(ids)
        size = len(self.segList.BUSList)
        print(f"segList size={size}")


    def initUI2(self):
        self.load((7,))
        seg = self.segList.BUSList[0]
        seg.findContours(addImages=True)
        singleUIObj = singleUI()
        singleUIObj.setSeg(seg)
        singleUIObj.setControlBox()
        tmp = []
        tmp.append(singleUIObj.getOutput())
        singleVBox = widgets.VBox(tmp)
        display(singleVBox)

    def initUI(self):
        singleUIObj = singleUI()
        singleUIObj.parent = self
        singleUIObj.initUI()
        tmp = []
        tmp.append(singleUIObj.getOutput())
        singleVBox = widgets.VBox(tmp)
        display(singleVBox)

        singleUIObj.initObserve()


class singleUI(object):

    logger = logging.getLogger("BUS." + __name__)


    def __init__(self):
        self.parent = None
        self.baseW = None
        self.seg = None
        self.segList = None
        self.idWList = None
        self.buttonLoad = None
        self.select_id = None
        self.imageWidth = None
        self.imageSelect = None
        self.imageWDisplay = None
        self.buttonApplyImgSelect = None
        self.qgrid1 = None
        self.compView = None

    def setSegList(self, seglist):
        self.segList = seglist
        # id = seg.id
    
    def setSeg(self, segment):
        self.seg = segment
        # id = seg.id
    
    def initUI(self):
        if self.seg is not None:
            id = self.seg.id
        else:
            id = 0
        
        self.idWList = widgets.Text(
            value="4-6",
            placeholder='',
            description='Enter id list:',
            disabled=False
        )
        
        self.buttonLoad = widgets.Button(
            description='Load',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='check' # (FontAwesome names without the `fa-` prefix)
        )
        
        self.select_id = widgets.Dropdown(
            options=['None'],
            value='None',
            description='Number:',
            disabled=False,
        )
        
        self.buttonPrev = widgets.Button(
            description='< Prev',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='check' # (FontAwesome names without the `fa-` prefix)
        )
        
        self.buttonNext = widgets.Button(
            description='> Next',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='check' # (FontAwesome names without the `fa-` prefix)
        )
        
        widthOpts = [300,400,500,600,700,800, 1600, 2400]
        self.imageWidth = widgets.Dropdown(
            options=widthOpts,
            value=widthOpts[0],
            description='Image width:',
            disabled=False
        )
        if self.seg is not None :
            opts = list(self.seg.images.keys())
        else:
            opts = ["None"]
        self.imageSelect = widgets.SelectMultiple(
            options=opts,
            value=opts,
            description='Images:',
            disabled=False,
        )

        self.buttonApplyImgSelect = widgets.Button(
            description='Apply',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='check', # (FontAwesome names without the `fa-` prefix)

        )
        self.buttonApplyImgSelect.layout.visibility = 'hidden'

    def initImageBoxes(self):
        imagesH = None
        imagesWList = []

        try:

            if self.seg is not None :
                opts = list(self.seg.images.keys())
                self.logger.debug("id=%s", self.seg.id)
            else:
                opts = []
            for opt in opts:
                titleW = widgets.Label(value=opt)
                freezeW = widgets.Checkbox(
                            value=False,
                            description='Freeze',
                            disabled=False,
                            indent=False
                        )
                imageBytes = self.getImageBytes(self.seg.images.get(opt).get('image'))
                imageW = widgets.Image(
                    value=imageBytes,
                    format='png',
                    width=self.imageWidth.value
                )
                imageVBox = widgets.VBox([titleW, freezeW, imageW])
                imagesWList.append(imageVBox)
            box_layout = widgets.Layout(overflow='scroll hidden',
                    border='3px solid black',
                    width='100%',
                    height='',
                    flex_flow='row nowrap',
                    display='flex')
            self.logger.debug("imagesWList=%s", imagesWList)
            if self.imageWDisplay is not None:
                self.imageWDisplay.children = imagesWList
            else:
                self.imageWDisplay = widgets.HBox(imagesWList, layout=box_layout)
            return( self.imageWDisplay)

        except:
            self.logger.exception("")
            self.exception(self.seg)
            raise

    def getImageBytes(self, img, addGrids=False):
        try:
            if addGrids :
                import matplotlib.pyplot as plt
                from io import BytesIO
                fig, ax = plt.subplots()
                # ax.title.set_text(key)
                plt.imshow(img, cmap="gray")
                figdata = BytesIO()
                fig.savefig(figdata, format='png')
                output = figdata.getvalue()
            else:
                is_success, im_buf_arr = cv2.imencode(".png", img)
                byte_im = im_buf_arr.tobytes()
                output = byte_im
            return output
        except:
            self.logger.excpetion("")
            self.exception(self.seg)
            raise

    def initDataFrameBoxes(self):
        # https://hub.gke2.mybinder.org/user/quantopian-qgrid-notebooks-lln1r8wy/notebooks/index.ipynb
        try:
            df = pd.read_csv('./data/dataScored.csv', header=0)
            # qgrid_widget = qgrid.show_grid(df, show_toolbar=False)
            # qgrid_widget.layout = widgets.Layout(width='100%')
            box_layout = widgets.Layout(overflow='scroll hidden',
                    border='3px solid black',
                    width='100%',
                    height='',
                    flex_flow='row nowrap',
                    display='flex')
            # datagrid = DataGrid(df, base_row_size=32, base_column_size=150)
            # datagrid = DataGrid(df, selection_mode="cell", editable=True)
            datagrid = DataGrid(df, layout={"height":"200px"})
            self.qgrid1 = widgets.HBox([datagrid], layout=box_layout)
            return(self.qgrid1)
        except:
            self.logger.exception("")
            raise

    def initComparisonView(self):
        compView = None
        try:

            compTitle = widgets.Label(value="Comparison View")
            if self.seg is not None :
                opts = list(self.seg.images.keys())
            else:
                opts = ["None"]
            compImageSelect = widgets.SelectMultiple(
                options=opts,
                value=opts,
                description='Images:',
                disabled=False,
            )
            widthOpts = [300,400,500,600,700,800, 1600, 2400]
            compImageWidth = widgets.Dropdown(
                options=widthOpts,
                value=widthOpts[0],
                description='Image width:',
                disabled=False
            )

            box_layout = widgets.Layout(overflow='scroll hidden',
                    border='3px solid black',
                    width='100%',
                    height='',
                    flex_flow='row nowrap',
                    display='flex')

            compControls = widgets.HBox([compTitle, compImageSelect, compImageWidth])
            compImageView = widgets.Hbox([])
            compView = widgets.VBox([compControls, compImageView], layout = box_layout)
            self.compView = compView
        except:
            self.logger.exception("")
            raise
        return(self.compView)

    def initCompImageView(self):
        imagesH = None
        imagesWList = []

        try:

            if self.seg is not None :
                opts = list(self.seg.images.keys())
                self.logger.debug("id=%s", self.seg.id)
            else:
                opts = []
            for opt in opts:
                titleW = widgets.Label(value=opt)
                freezeW = widgets.Checkbox(
                            value=False,
                            description='Freeze',
                            disabled=False,
                            indent=False
                        )
                imageBytes = self.getImageBytes(self.seg.images.get(opt).get('image'))
                imageW = widgets.Image(
                    value=imageBytes,
                    format='png',
                    width=self.imageWidth.value
                )
                imageVBox = widgets.VBox([titleW, freezeW, imageW])
                imagesWList.append(imageVBox)
            box_layout = widgets.Layout(overflow='scroll hidden',
                    border='3px solid black',
                    width='100%',
                    height='',
                    flex_flow='row nowrap',
                    display='flex')
            self.logger.debug("imagesWList=%s", imagesWList)
            if self.imageWDisplay is not None:
                self.imageWDisplay.children = imagesWList
            else:
                self.imageWDisplay = widgets.VBox(imagesWList, layout=box_layout)
            return( self.imageWDisplay)

        except:
            self.logger.exception("")
            self.error(self.seg)
            raise



    def initObserve(self):
        self.imageSelect.observe(self.on_imageSelect_change, names='value')
        # imageWidth.observe(on_width_change, names='value')
        self.buttonLoad.on_click(self.on_load_clicked)
        self.buttonPrev.on_click(self.on_prev_clicked)
        self.buttonNext.on_click(self.on_next_clicked)
        self.select_id.observe(self.on_select_id_change, names='value')
        # self.buttonApplyImgSelect(self.on_image_select_change, names='value')

    def on_load_clicked(self, b):
        self.logger.debug(b)
        breakpoint()
        listString = self.idWList.value
        result = self.valid_id_string(listString)
        self.logger.debug(result)
        if result[0] :
            self.select_id.options = result[2]
            self.select_id.value = result[2][0]

    def on_next_clicked(self, b):
        self.logger.debug(b)
        options = self.select_id.options
        size = len(options)
        idxList = [i for i, value in enumerate(self.select_id.options) if value == self.select_id.value]
        idx = idxList[0]
        idx = idx + 1
        if idx == size :
            idx = 0
        self.select_id.value = options[idx]

    def on_prev_clicked(self, b):
        self.logger.debug(b)
        options = self.select_id.options
        size = len(options)
        idxList = [i for i, value in enumerate(self.select_id.options) if value == self.select_id.value]
        idx = idxList[0]
        idx = idx - 1
        if idx == -1 :
            idx = size - 1
        self.select_id.value = options[idx]

    def on_select_id_change(self, change):
        self.logger.debug(change)
        new_value = change['new']
        self.logger.debug("new_value=%s", new_value)
        ids = (int(new_value),)
        self.load(ids)
        if self.seg is not None :
            opts = list(self.seg.images.keys())
        else:
            opts = ["None"]
        self.logger.debug("opts=%s", opts)
        self.imageSelect.options = opts
        self.imageSelect.value = opts
        self.initImageBoxes()

    def on_imageSelect_change(self, change):
        self.logger.debug(change)
        new_value = change['new']
        self.logger.debug("new_value=%s", new_value)
        self.buttonApplyImgSelect.layout.visibility = 'visible'
        # ids = (int(new_value),)
        # self.load(ids)
        # if self.seg is not None :
        #     opts = list(self.seg.images.keys())
        # else:
        #     opts = ["None"]
        # self.logger.debug(opts)
        # self.imageSelect.options = opts
        # self.imageSelect.value = opts



    def load(self, ids):
        if self.segList is not None:
            self.logger.debug("id()=%s", id(self.segList))
        self.segList = BUSSegmentorList()
        self.logger.debug("id()=%s", id(self.segList))
        self.segList.loadDataSetB(ids)
        size = len(self.segList.BUSList)
        self.logger.debug(f"segList size={size}")
        self.seg = self.segList.BUSList[size-1]
        self.seg.findContours(addImages=True)

    def valid_id_string(self, idString):
        message = ""
        idList = []
        success = True
        try:
            rangeList = idString.split(",")
            for rg in rangeList:
                if '-' in rg:
                    range2 = rg.split('-')
                    start = int(range2[0])
                    end = int(range2[1]) + 1
                    idList.extend(range(start, end))
                else:
                    idList.append(int(rg))
        except:
            success = False
        return((success, message,  idList))

    def getOutput(self):
        box_layout = widgets.Layout(overflow='scroll hidden',
                    border='3px solid black',
                    width='100%',
                    height='',
                    flex_flow='row wrap',
                    display='flex')
        controlW = widgets.HBox([self.idWList, self.buttonLoad, self.select_id, self.buttonPrev, self.buttonNext, self.imageWidth,
                self.imageSelect, self.buttonApplyImgSelect], layout=box_layout)
        output = widgets.VBox([controlW, self.initImageBoxes(), self.initDataFramePanel()])
        self.baseW = output
        return(output)





