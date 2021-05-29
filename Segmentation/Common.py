from pathlib import Path

class Common(object):

    HomeProjectFolder = None

    @staticmethod
    def getImagePath():
        path = Common.getHomeProjectPath() / "Datasets" / "BUS_Dataset_B"
        return path

    @staticmethod
    def getHomeProjectPath():
        path = Path(Common.HomeProjectFolder)
        return path