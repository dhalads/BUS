import json
from types import SimpleNamespace
import logging
import pandas as pd

class BUSDataTable(object):
    
    logger = logging.getLogger("BUS." + __name__)

    def __init__(self):
        self.name = None
        self.source = None
        self.queryString = None
        self.sortList = None
        self.sortAscendingList = None
        self.columnList = None
        self.isFollowImageList = False
        self.isFollowSingle = False

    def save(self):
        try:
            filename = 'data/' + self.name + '.json'
            # https://www.geeksforgeeks.org/encoding-and-decoding-custom-objects-in-python-json/
            output = json.dumps(self, indent = 4, default=lambda o: o.__dict__)
            with open(filename, "w") as outfile:
                outfile.write(json_object)
        except:
            pass

    def load(self):
        try:
            filename = 'data/' + self.name + '.json'
            # https://stackoverflow.com/questions/6578986/how-to-convert-json-data-into-a-python-object
            with open(filename, 'r') as openfile:
                # Reading from json file
                json_object = json.load(openfile)
        except:
            self.logger.exception("")
            raise

    def createDF(self):
        try:

        except:

