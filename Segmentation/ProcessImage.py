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
from BUSDataTable import BUSDataTable
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
import fnmatch
import os
import ast
# from ipydatagrid import DataGrid
# from beakerx import *
# from beakerx.object import beakerx
# from google.colab import data_table
# https://github.com/man-group/dtale#google-colab
# import dtale
# import dtale.app as dtale_app


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
        import logging
        import json, logging.config

        from IPython.core.display import display, HTML
        display(HTML("<style>.container { width:100% !important; }</style>"))

        if "google.colab" in sys.modules:
            # from google.colab import drive
            # drive.mount('/content/gdrive')
            # os.chdir('/content/gdrive/.shortcut-targets-by-id/1vMzK9J4qysmceXoMlEQPYZoLdu67Wu64/BUS Project Home/Share_with_group/David_Halama/BUS')
            # sys.path.append('/content/gdrive/.shortcut-targets-by-id/1vMzK9J4qysmceXoMlEQPYZoLdu67Wu64/BUS Project Home/Share_with_group/David_Halama/BUS/Segmentation')
            # from Common import Common
            # Common.HomeProjectFolder = "/content/gdrive/.shortcut-targets-by-id/1vMzK9J4qysmceXoMlEQPYZoLdu67Wu64/BUS Project Home"
            # print(os.getcwd())
            # !ls -la /content/gdrive/MyDrive
            os.chdir('/home/djhalama/Documents/GitHub/BUS')
            sys.path.append('/home/djhalama/Documents/GitHub/BUS/Segmentation')
            from Common import Common
            Common.HomeProjectFolder = "/mnt/c/Users/djhalama/Documents/Education/DS-785/BUS Project Home"
            with open('logging-config-colab.json', 'rt') as f:
                config = json.load(f)
                logging.config.dictConfig(config)
        else:
            os.chdir('/home/djhalama/Documents/GitHub/BUS')
            sys.path.append('/home/djhalama/Documents/GitHub/BUS/Segmentation')
            from Common import Common
            Common.HomeProjectFolder = "/mnt/c/Users/djhalama/Documents/Education/DS-785/BUS Project Home"
            with open('logging-config.json', 'rt') as f:
                config = json.load(f)
                logging.config.dictConfig(config)

class busUI(object):

    logger = logging.getLogger("BUS." + __name__)

    def __init__(self):
        self.segList = None
        self.UI = None

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
        # tmp = []
        # tmp.append(singleUIObj.getOutput())
        # singleVBox = widgets.VBox(tmp)
        # self.UI = singleVBox
        # display(singleVBox)
        display(singleUIObj.getOutput())
        singleUIObj.initObserve()


class singleUI(object):

    logger = logging.getLogger("BUS." + __name__)


    def __init__(self):
        self.parent = None
        self.baseW = None
        self.segList = None
        self.seg = None
        self.freezeSegList = None
        self.idWList = None
        self.buttonLoad = None
        self.select_id = None
        self.imageWidth = None
        self.imageSelect = None
        self.imageWDisplay = None
        self.buttonApplyImgSelect = None
        self.singleSelectedImages = None
        self.gridPanel = None
        self.compView = None
        self.compControls = None
        self.compImageView = None
        self.select_num_to_display = None
        self.freezeList = []
        self.displayOrientation = None

        self.gridWidgets = {}
        self.gridList = []

    def setSegList(self, seglist):
        self.segList = seglist
        # id = seg.id

    # def setSeg(self, segment):
    #     self.seg = segment
    #     # id = seg.id
    
    def initUI(self):
        # if self.seg is not None:
        #     id = self.seg.id
        # else:
        #     id = 0
        
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
            options=[],
            value=None,
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

        self.select_num_to_display = widgets.Dropdown(
            options=['1', '2', '3', '4', '5'],
            value='1',
            description='Number to Display:',
            disabled=False,
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
            opts = ["Original"]
        self.imageSelect = widgets.SelectMultiple(
            options=opts,
            value=opts,
            description='Images:',
            disabled=False,
        )

        self.displayOrientation = widgets.RadioButtons(
                options=['vertical', 'horizontal'],
                value='vertical',
            #    layout={'width': 'max-content'}, # If the items' names are long
                description='Orientation',
                disabled=False
            )



        self.buttonApplyImgSelect = widgets.Button(
            description='Apply',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Click me',
            icon='check', # (FontAwesome names without the `fa-` prefix)

        )
        # self.buttonApplyImgSelect.layout.visibility = 'hidden'

    def initGridUI(self):
        try:
            tableOpts = self.getSourceList()
            source = widgets.Dropdown(
                    options=tableOpts,
                    value=None,
                    description='available:',
                    disabled=False
                )

            self.gridWidgets["grid.source"] = source

            buttonSourceAdd = widgets.Button(
                    description='Add',
                    disabled=False,
                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                    tooltip='Click me',
                    icon='check' # (FontAwesome names without the `fa-` prefix)
                )

            self.gridWidgets["grid.buttonSourceAdd"] = buttonSourceAdd

            message = widgets.Text(
                value='',
                placeholder='Type something',
                description='Message:',
                disabled=False,
                layout=widgets.Layout(width='100%')
            )

            self.gridWidgets["grid.message"] = message

            sourceBox = widgets.HBox([source, buttonSourceAdd, message])

            loaded = widgets.Dropdown(
                    options=[],
                    value=None,
                    description='loaded:',
                    disabled=False
                )
            
            self.gridWidgets["grid.loaded"] = loaded
            self.setGridLoaded(None)

            selectFollow = widgets.Dropdown(
                    options=['None', 'ImageList', 'Single'],
                    value=None,
                    description='follow:',
                    disabled=False
                )
            
            self.gridWidgets["grid.selectFollow"] = selectFollow

            buttonLoadedUpdate = widgets.Button(
                    description='Update',
                    disabled=False,
                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                    tooltip='Click me',
                    icon='check' # (FontAwesome names without the `fa-` prefix)
                )

            self.gridWidgets["grid.buttonLoadedUpdate"] = buttonLoadedUpdate

            buttonLoadedLoad = widgets.Button(
                    description='Load',
                    disabled=False,
                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                    tooltip='Click me',
                    icon='check' # (FontAwesome names without the `fa-` prefix)
                )

            self.gridWidgets["grid.buttonLoadedLoad"] = buttonLoadedLoad

            buttonLoadedSave = widgets.Button(
                    description='Save',
                    disabled=False,
                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                    tooltip='Click me',
                    icon='check' # (FontAwesome names without the `fa-` prefix)
                )

            self.gridWidgets["grid.buttonLoadedSave"] = buttonLoadedSave

            buttonLoadedRemove = widgets.Button(
                    description='Remove',
                    disabled=False,
                    button_style='', # 'success', 'info', 'warning', 'danger' or ''
                    tooltip='Click me',
                    icon='check' # (FontAwesome names without the `fa-` prefix)
                )

            self.gridWidgets["grid.buttonLoadedRemove"] = buttonLoadedRemove

            tableBox = widgets.HBox([loaded, selectFollow, buttonLoadedUpdate, buttonLoadedLoad, buttonLoadedSave ,buttonLoadedRemove])

            name = widgets.Text(
                value='',
                placeholder='Type something',
                description='name:',
                disabled=False,
                layout=widgets.Layout(width='100%')
            )

            self.gridWidgets["grid.name"] = name

            sourceText = widgets.Text(
                value='',
                placeholder='Type something',
                description='source:',
                disabled=False,
                layout=widgets.Layout(width='100%')
            )

            self.gridWidgets["grid.sourceText"] = sourceText
            
            queryString = widgets.Text(
                value='',
                placeholder='Type something',
                description='Query String:',
                disabled=False,
                layout=widgets.Layout(width='100%')
            )

            self.gridWidgets["grid.queryString"] = queryString

            sortList = widgets.Text(
                value='',
                placeholder='Type something',
                description='Sort List:',
                disabled=False,
                layout=widgets.Layout(width='100%')
            )

            self.gridWidgets["grid.sortList"] = sortList

            sortAssendingList = widgets.Text(
                value='',
                placeholder='Type something',
                description='Sort Assending List:',
                disabled=False,
                layout=widgets.Layout(width='100%')
            )

            self.gridWidgets["grid.sortAscendingList"] = sortAssendingList

            columnList = widgets.Text(
                value='',
                placeholder='Type something',
                description='Column List:',
                disabled=False,
                layout=widgets.Layout(width='100%')
            )

            self.gridWidgets["grid.columnList"] = columnList

            box_layout = widgets.Layout(overflow='scroll hidden',
                    border='3px solid black',
                    width='100%',
                    height='',
                    flex_flow='column',
                    display='flex')


            self.gridWidgets["grid.buttonSourceAdd"].on_click(self.on_grid_add_clicked)
            self.gridWidgets["grid.buttonLoadedUpdate"].on_click(self.on_grid_update_clicked)
            self.gridWidgets["grid.buttonLoadedLoad"].on_click(self.on_grid_load_clicked)
            self.gridWidgets["grid.buttonLoadedSave"].on_click(self.on_grid_save_clicked)
            self.gridWidgets["grid.buttonLoadedRemove"].on_click(self.on_grid_remove_clicked)
            self.gridWidgets["grid.loaded"].observe(self.on_grid_loaded_change, names='value')
            

            output = widgets.VBox([sourceBox, tableBox, name, sourceText, queryString, sortList, sortAssendingList, columnList], layout=box_layout)

        except:
            self.logger.exception("")
            raise
        return output

    def initImageBoxes(self):
        imagesH = None
        imagesWList = []

        try:

            if self.seg is not None :
                opts = self.imageSelect.value
            else:
                opts = []
            for opt in opts:
                titleW = widgets.Label(value=opt)
                # freezeW = widgets.Checkbox(
                #             value=False,
                #             description='Freeze',
                #             disabled=False,
                #             indent=False
                #         )
                imageBytes = self.getImageBytes(self.seg.images.get(opt).get('image'))
                imageW = widgets.Image(
                    value=imageBytes,
                    format='png',
                    width=self.imageWidth.value
                )
                imageVBox = widgets.VBox([titleW, imageW])
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

    def initDataFramePanel(self):
        # https://hub.gke2.mybinder.org/user/quantopian-qgrid-notebooks-lln1r8wy/notebooks/index.ipynb
        try:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            # pd.set_option('display.width', None)
            # pd.set_option('display.max_colwidth', None)

            box_layout = widgets.Layout(overflow='scroll',
                    # border='3px solid black',
                    width='100%',
                    # height='300px',
                    # flex_flow='row nowrap',
                    display='flex')

            self.gridPanel = widgets.VBox([], layout=box_layout)
            return(self.gridPanel)
        except:
            self.logger.exception("")
            raise

    def initGridListPanel(self):
        # https://hub.gke2.mybinder.org/user/quantopian-qgrid-notebooks-lln1r8wy/notebooks/index.ipynb
        try:
            pd.set_option('display.max_rows', None)
            pd.set_option('display.max_columns', None)
            # pd.set_option('display.width', None)
            # pd.set_option('display.max_colwidth', None)
            # box_layout = widgets.Layout(overflow='scroll',
            #         # border='3px solid black',
            #         width='100%',
            #         height='300px',
            #         flex_flow='row nowrap',
            #         display='flex')
            wList = []
            value = self.select_id.value
            if value is not None:
                id = int(value)
                idList = [x.id for x in self.getCompImageList()]
            else:
                id = None
                idList = None
            for grid in self.gridList:
                wList.append(grid.get_panel(id, idList))
            self.gridPanel.children = wList
        except:
            self.logger.exception("")
            raise

    def initComparisonView(self):

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

            self.compControls = widgets.HBox([compTitle, compImageSelect, compImageWidth])
            self.compImageView = widgets.GridBox()
            self.compView = widgets.VBox([self.compImageView], layout = box_layout)
        except:
            self.logger.exception("")
            raise
        return(self.compView)

    def initCompImageView3(self):
        imagesH = None
        imagesWList = []
        imageBox = None

        try:
            legendList = []
            gridarea = str(1) + "-" + str(1)
            idLegend = widgets.Label(value="id->", layout=widgets.Layout(grid_area=gridarea, grid_column="1 / 1",grid_row="1 / 1" ))
            legendList.append(idLegend)
            if self.segList is not None :
                opts = self.imageSelect.value
            else:
                opts = []
            # if self.seg is not None :
            #     opts = list(self.seg.images.keys())
            #     self.logger.debug("id=%s", self.seg.id)
            # else:
            #     opts = []
            for idxopt, opt in enumerate(opts):
                gridarea = str(1) + "-" + str(idxopt + 3)
                gridrow = str(idxopt + 2) + " / " + str(idxopt + 2)
                imageNameLegend = widgets.Label(value=opt, layout=widgets.Layout(grid_area=gridarea, grid_column="1 / 1",grid_row=gridrow, width = 'max-content'))
                legendList.append(imageNameLegend)
            # legendVBox = widgets.VBox(legendList)
            # imagesWList.append(legendVBox)
            imagesWList.extend(legendList)



            if self.segList is not None :
                images = self.getCompImageList()
            else:
                images = []

            for idximg, img in enumerate(images):
                VList = []
                gridcolumn = str(idximg + 2) + " / " + str(idximg + 2)
                gridarea = str(idximg + 2) + "-" + str(1)
                titleW = widgets.Label(value=str(img.id)
                # layout=widgets.Layout(grid_area=gridarea, grid_column=gridcolumn,grid_row="1 / 1")
                )
                # VList.append(titleW)
                gridarea = str(idximg + 2) + "-" + str(2)
                if img.id in self.freezeList:
                    freezeValue = True
                else:
                    freezeValue = False
                freezeW = widgets.Checkbox(
                            value=freezeValue,
                            description='Freeze -> ' + str(img.id),
                            disabled=False,
                            indent=False
                            # layout=widgets.Layout(grid_area=gridarea, grid_column=gridcolumn,grid_row="1 / 1")
                        )
                freezeW.observe(self.on_freeze_change, names='value')
                topgrid = widgets.HBox([titleW, freezeW], layout=widgets.Layout(grid_area=gridarea, grid_column=gridcolumn,grid_row="1 / 1"))
                VList.append(topgrid)
                for idxopt, opt in enumerate(opts):
                    imageBytes = self.getImageBytes(img.images.get(opt).get('image'))
                    gridrow = str(idxopt+2) + " / " + str(idxopt+2)
                    gridarea = str(idximg + 2) + "-" + str(idxopt+3)
                    imageW = widgets.Image(
                        value=imageBytes,
                        format='png',
                        width=self.imageWidth.value,
                        layout=widgets.Layout(grid_area=gridarea, grid_column=gridcolumn,grid_row=gridrow)
                    )
                    VList.append(imageW)
                # imageVBox = widgets.VBox(VList)
                # imagesWList.append(imageVBox)
                imagesWList.extend(VList)
            box_layout = widgets.Layout(overflow='scroll hidden',
                                        border='3px solid black',
                                        width='100%',
                                        height='',
                                        flex_flow='row nowrap',
                                        display='flex')
            gridWidth = len(opts)
            gridLength = len(images)
            gridlayout=widgets.Layout(width='100%',
                                        grid_template_columns='max-content max-content max-content max-content max-content max-content',
                                        grid_template_rows='auto auto auto auto auto auto auto',
                                        grid_gap='5px 5px',
                                        grid_template_areas='''
                                            "1-1 2-1 3-1 4-1"
                                            "1-2 2-2 3-2 4-2"
                                            "1-3 2-3 3-3 4-3"
                                            ''')
            self.logger.debug("imagesWList=%s", imagesWList)
            if self.compImageView is not None:
                self.compImageView.layout = gridlayout
                self.compImageView.children = imagesWList
            else:
                self.compImageView.layout = gridlayout
                self.compImageView.children = imagesWList
            self.logger.debug("compImageView=%s", self.compImageView)
            return( self.compImageView)

        except:
            self.logger.exception("")
            self.error(self.segList)
            raise

    def initCompImageView(self):
        imagesH = None
        imagesWList = []
        imageBox = None

        try:
            if self.displayOrientation.value == 'vertical':
                isHorizontal = False
            else:
                isHorizontal = True
            legendList = []
            if isHorizontal:
                rownum = 1
                colnum = 1
            else:
                rownum = 1
                colnum = 1
            gridarea, gridcolumn, gridrow = self.getLayoutGrid(colnum, rownum)
            gridarea = str(1) + "-" + str(1)
            idLegend = widgets.Label(value="id->", layout=widgets.Layout(grid_area=gridarea, grid_column=gridcolumn,grid_row=gridrow ))
            legendList.append(idLegend)
            if self.segList is not None :
                opts = self.imageSelect.value
            else:
                opts = []
            # if self.seg is not None :
            #     opts = list(self.seg.images.keys())
            #     self.logger.debug("id=%s", self.seg.id)
            # else:
            #     opts = []
            for idxopt, opt in enumerate(opts):
                if isHorizontal:
                    rownum = 1
                    colnum = idxopt + 2
                else:
                    rownum = idxopt + 2
                    colnum = 1
                gridarea, gridcolumn, gridrow = self.getLayoutGrid(colnum, rownum)
                imageNameLegend = widgets.Label(value=opt, layout=widgets.Layout(grid_area=gridarea, grid_column=gridcolumn,grid_row=gridrow, width = 'max-content'))
                legendList.append(imageNameLegend)
            # legendVBox = widgets.VBox(legendList)
            # imagesWList.append(legendVBox)
            imagesWList.extend(legendList)



            if self.segList is not None :
                images = self.getCompImageList()
            else:
                images = []

            for idximg, img in enumerate(images):
                VList = []
                titleW = widgets.Label(value=str(img.id)
                # layout=widgets.Layout(grid_area=gridarea, grid_column=gridcolumn,grid_row="1 / 1")
                )
                # VList.append(titleW)
                if img.id in self.freezeList:
                    freezeValue = True
                else:
                    freezeValue = False
                freezeW = widgets.Checkbox(
                            value=freezeValue,
                            description='Freeze -> ' + str(img.id),
                            disabled=False,
                            indent=False
                            # layout=widgets.Layout(grid_area=gridarea, grid_column=gridcolumn,grid_row="1 / 1")
                        )
                freezeW.observe(self.on_freeze_change, names='value')
                if isHorizontal:
                    rownum = idximg + 2
                    colnum = 1
                else:
                    rownum = 1
                    colnum = idximg + 2
                gridarea, gridcolumn, gridrow = self.getLayoutGrid(colnum, rownum)
                topgrid = widgets.HBox([titleW, freezeW], layout=widgets.Layout(grid_area=gridarea, grid_column=gridcolumn,grid_row=gridrow))
                VList.append(topgrid)
                for idxopt, opt in enumerate(opts):
                    imageBytes = self.getImageBytes(img.images.get(opt).get('image'))
                    if isHorizontal:
                        rownum = idximg + 2
                        colnum = idxopt + 2
                    else:
                        rownum = idxopt + 2
                        colnum = idximg + 2
                    gridarea, gridcolumn, gridrow = self.getLayoutGrid(colnum, rownum)
                    imageW = widgets.Image(
                        value=imageBytes,
                        format='png',
                        width=self.imageWidth.value,
                        layout=widgets.Layout(grid_area=gridarea, grid_column=gridcolumn,grid_row=gridrow)
                    )
                    VList.append(imageW)
                # imageVBox = widgets.VBox(VList)
                # imagesWList.append(imageVBox)
                imagesWList.extend(VList)
            box_layout = widgets.Layout(overflow='scroll hidden',
                                        border='3px solid black',
                                        width='100%',
                                        height='',
                                        flex_flow='row nowrap',
                                        display='flex')
            gridWidth = len(opts)
            gridLength = len(images)
            gridlayout=widgets.Layout(width='100%',
                                        grid_template_columns='100px max-content max-content max-content max-content max-content max-content max-content',
                                        grid_template_rows='auto auto auto auto auto auto auto auto auto',
                                        grid_gap='5px 5px',
                                        grid_template_areas='''
                                            "1-1 2-1 3-1 4-1"
                                            "1-2 2-2 3-2 4-2"
                                            "1-3 2-3 3-3 4-3"
                                            ''')
            self.logger.debug("imagesWList=%s", imagesWList)
            if self.compImageView is not None:
                self.compImageView.layout = gridlayout
                self.compImageView.children = imagesWList
            else:
                self.compImageView.layout = gridlayout
                self.compImageView.children = imagesWList
            self.logger.debug("compImageView=%s", self.compImageView)
            return( self.compImageView)

        except:
            self.logger.exception("")
            self.logger.error("segList=%s", str(self.segList))
            raise

    def getLayoutGrid(self, colnum, rownum):
        gridarea = str(colnum) + "-" + str(rownum)
        gridcolumn = str(colnum) + " / " + str(colnum)
        gridrow = str(rownum) + " / " + str(rownum)
        return((gridarea, gridcolumn, gridrow))

    def getCompImageList(self):
        output = []
        try:
            idList = []
            idList.extend(self.freezeList)
            numToDisplay = int(self.select_num_to_display.value)
            options = self.select_id.options
            size = len(options)
            idxList = [i for i, value in enumerate(self.select_id.options) if value == self.select_id.value]
            idx = idxList[0]

            for i in range(0, len(options)):
                if len(idList) >= numToDisplay:
                    break
                id = int(options[idx])
                if not id in idList:
                    idList.append(id)
                idx = idx + 1
                if idx == size :
                    idx = 0
            self.logger.debug("idList=%s", idList)
            self.load(idList)
            for id in idList:
                output.extend([x for x in self.segList.BUSList if x.id == id])
            self.logger.debug("output=%s", output)
        except:
            self.logger.exception("")
            raise
        return(output)



    def initObserve(self):
        self.imageSelect.observe(self.on_imageSelect_change, names='value')
        self.buttonApplyImgSelect.on_click(self.on_applyImgSelect_clicked)
        # imageWidth.observe(on_width_change, names='value')
        self.buttonLoad.on_click(self.on_load_clicked)
        self.buttonPrev.on_click(self.on_prev_clicked)
        self.buttonNext.on_click(self.on_next_clicked)
        self.select_id.observe(self.on_select_id_change, names='value')
        # self.buttonApplyImgSelect(self.on_image_select_change, names='value')

    def on_load_clicked(self, b):
        self.logger.debug(b)
        listString = self.idWList.value
        result = self.valid_id_string(listString)
        self.logger.debug(result)
        if result[0] :
            self.select_id.options = result[2]
            self.select_id.value = result[2][0]

    def on_next_clicked(self, b):
        self.logger.debug(b)
        self.logger.debug("dynamic code reload 2")
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
        loadImageSelect = False
        new_value = change['new']
        self.logger.debug("new_value=%s", new_value)
        ids = (int(new_value),)
        if self.segList is None:
            loadImageSelect = True
        self.load(ids)
        if loadImageSelect :
            opts = list(self.segList.BUSList[0].images.keys())
            self.imageSelect.options = opts
            self.imageSelect.value = ["Original"]
            self.logger.debug("opts=%s", opts)

        # self.initImageBoxes()
        self.initCompImageView()
        self.initGridListPanel()

    def on_imageSelect_change(self, change):
        self.logger.debug(change)
        new_value = change['new']
        old_value = change['old']
        self.logger.debug("new_value=%s", new_value)
        self.logger.debug("old_value=%s", old_value)
        if len(old_value) != 0 :
            # self.buttonApplyImgSelect.layout.visibility = 'visible'
            pass
        # ids = (int(new_value),)
        # self.load(ids)
        # if self.seg is not None :
        #     opts = list(self.seg.images.keys())
        # else:
        #     opts = ["None"]
        # self.logger.debug(opts)
        # self.imageSelect.options = opts
        # self.imageSelect.value = opts

    def on_applyImgSelect_clicked(self, b):
        self.logger.debug("b=%s", b)
        self.singleSelectedImages = list(self.imageSelect.value).copy()
        self.logger.debug("self.singleSelectedImages=%s", self.singleSelectedImages)
        # self.initImageBoxes()
        self.initCompImageView()
        self.initGridListPanel()
        # self.buttonApplyImgSelect.layout.visibility = 'hidden'

    def on_freeze_change(self, change):
        try:
            self.logger.debug("change=%s", change)
            owner = change['owner']
            new_value = change['new']
            id =  int(owner.description.split(">")[1].strip())
            self.logger.debug("owner=%s", owner)
            self.logger.debug("new_value=%s", new_value)
            self.logger.debug("id=%s", id)
            if new_value :
                self.freezeList.append(id)
            else:
                self.freezeList.remove(id)
            self.logger.debug("freezeList=%s", self.freezeList)
        except:
            self.logger.exception("")
            raise

    def on_grid_add_clicked(self, b):
        try:
            self.logger.debug("b=%s", b)
            source = self.gridWidgets["grid.source"].value
            busDT = BUSDataTable()
            if source.endswith(".csv") :
                busDT.name = 'New.json'
                busDT.source = source
                busDT.queryString = ''
                busDT.sortList = []
                busDT.sortAscendingList = []
                busDT.columnList = []
            else:
                busDT.name = source
                busDT.selectFollow = 'None'
                busDT.load()

                # ",".join(map(str, idList))

            self.logger.debug("busDT=%s", str(busDT))
            self.gridList.append(busDT)
            idx = len(self.gridList) - 1
            self.logger.debug("gridList=%s", str(self.gridList))
            self.setGridLoaded(idx)
            busDT.init_df()
            self.initGridListPanel()
        except:
            self.logger.exception("")
            raise

    def on_grid_update_clicked(self, b):
        try:
            self.logger.debug("b=%s", b)
            self.gridWidgets['grid.message'].value = ''
            value = self.gridWidgets["grid.loaded"].value
            busDT = self.gridList[value]
            newDT = BUSDataTable()
            newDT.name = self.gridWidgets["grid.name"].value
            newDT.source = busDT.source
            newDT.queryString = self.gridWidgets["grid.queryString"].value
            newDT.sortList = ast.literal_eval(self.gridWidgets["grid.sortList"].value)
            newDT.sortAscendingList = ast.literal_eval(self.gridWidgets["grid.sortAscendingList"].value)
            newDT.columnList = ast.literal_eval(self.gridWidgets["grid.columnList"].value)
            newDT.selectFollow = self.gridWidgets["grid.selectFollow"].value
            self.logger.debug("newDT=%s", str(newDT))
            success, message = newDT.init_df()
            if success:
                self.gridList[value] = newDT
                self.initGridListPanel()
                self.gridWidgets['grid.message'].value = 'Success:' + message
            else:
                self.gridWidgets['grid.message'].value = 'Failed:' + message

        except Exception as e:
            self.logger.exception("")
            try:
                self.gridWidgets['grid.message'].value = 'Failed:'
            except:
                self.logger.exception("")
                raise
        return success

    def on_grid_load_clicked(self, b):
        try:
            self.logger.debug("b=%s", b)
            self.gridWidgets['grid.message'].value = ''
            value = self.gridWidgets["grid.loaded"].value
            busDT = self.gridList[value]
            idList = busDT.get_id_list()
            x = ",".join(map(str, idList))
            self.logger.debug("x=%s", str(x))
            self.idWList.value =x
            self.on_load_clicked(b)

        except Exception as e:
            self.logger.exception("")
            raise
    
    def on_grid_save_clicked(self, b):
        try:
            self.logger.debug("b=%s", b)
            self.gridWidgets['grid.message'].value = ''
            value = self.gridWidgets["grid.loaded"].value
            success = self.on_grid_update_clicked(b)
            busDT = self.gridList[value]
            if success:
                busDT.save()
                self.gridWidgets['grid.message'].value = 'Success:Save'
            else:
                # self.gridWidgets['grid.message'].value = 'Failed:'
                pass

        except Exception as e:
            self.logger.exception("")
            raise

    def on_grid_remove_clicked(self, b):
        try:
            self.logger.debug("b=%s", b)
            self.gridWidgets['grid.message'].value = ''
            value = self.gridWidgets["grid.loaded"].value
            if value is not None:
                busDT = self.gridList[value]
                del self.gridList[value]
                self.setGridLoaded(None)
                self.initGridListPanel()

        except Exception as e:
            self.logger.exception("")
            raise

    def setGridLoaded(self, value):
        try:
            sourceWidget = self.gridWidgets["grid.loaded"]
            options = [(val.name, i) for i, val in enumerate(self.gridList)]
            sourceWidget.options = options
            sourceWidget.value = value
            self.logger.debug("options=%s", str(options))
        except:
            self.logger.exception("")
            raise
    
    def on_grid_loaded_change(self, change):
        try:
            self.logger.debug("change=%s", change)
            owner = change['owner']
            new_value = change['new']
            if new_value is not None:
                busDT = self.gridList[new_value]
                self.gridWidgets["grid.name"].value = busDT.name
                self.gridWidgets["grid.sourceText"].value = busDT.source
                self.gridWidgets["grid.queryString"].value = busDT.queryString
                self.gridWidgets["grid.sortList"].value = str(busDT.sortList)
                self.gridWidgets["grid.sortAscendingList"].value = str(busDT.sortAscendingList)
                self.gridWidgets["grid.columnList"].value = str(busDT.columnList)
                self.gridWidgets["grid.selectFollow"].value = str(busDT.selectFollow)
                self.logger.debug("busDT=%s", str(busDT))
            else:
                busDT = None
                self.gridWidgets["grid.name"].value = ""
                self.gridWidgets["grid.sourceText"].value = ""
                self.gridWidgets["grid.queryString"].value = ""
                self.gridWidgets["grid.sortList"].value = str([])
                self.gridWidgets["grid.sortAscendingList"].value = str([])
                self.gridWidgets["grid.columnList"].value = str([])
                self.gridWidgets["grid.selectFollow"].value = 'None'
        except:
            self.logger.exception("busDT=%s", str(busDT))
            raise

    def load(self, ids):
        if self.segList is None:
            self.segList = BUSSegmentorList()
        self.segList.loadDataSetB(ids)
        size = len(self.segList.BUSList)
        self.logger.debug(f"segList size={size}")
        self.seg = self.segList.BUSList[size-1]
        # self.seg.findContours(addImages=True)

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

    def getSourceList(self):
        sourceList = None
        try:
            dirname = "data"
            pattern = "*.json"
            sourceList = fnmatch.filter(os.listdir(dirname), pattern)
            # pattern = "*.csv"
            # sourceList.extend(fnmatch.filter(os.listdir(dirname), pattern))
            sourceList.sort()
        except:
            self.logger.exception("")
            raise
        return sourceList

    def getOutput(self):
        box_layout = widgets.Layout(overflow='scroll hidden',
                    border='3px solid black',
                    width='100%',
                    height='',
                    flex_flow='row wrap',
                    display='flex')
        controlW = widgets.HBox([self.idWList, self.buttonLoad, self.select_id, self.buttonPrev, self.buttonNext, self.select_num_to_display, self.imageWidth,
                self.imageSelect, self.displayOrientation, self.buttonApplyImgSelect], layout=box_layout)
        box_layout = widgets.Layout(overflow='scroll',
                    border='3px solid black',
                    width='100%',
                    height='100%'
                    # flex_flow='row wrap',
                    # display='flex'
                    )
        output = widgets.VBox([self.initGridUI(), controlW, self.initComparisonView(), self.initDataFramePanel()], layout=box_layout)
        
        self.baseW = output
        return(output)










