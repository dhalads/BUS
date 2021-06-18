import unittest
import logging
import json, logging.config
import sys
sys.path.append('Segmentation')
sys.path.append('tests')
import BUSSegmentor
import BUSSegmentorList
from BUSDataTable import BUSDataTable

class BUSDataTable_test(unittest.TestCase):

    logger = logging.getLogger("BUS." + __name__)
    
    def setUp(self):
        with open('logging-config.json', 'rt') as f:
            config = json.load(f)
            logging.config.dictConfig(config)

    def test_Save(self):
        try:
            busDT = BUSDataTable()
            busDT.name = 'New'
            busDT.source = 'data/dataScored.csv'
            busDT.queryString = ''
            busDT.sortList = []
            busDT.sortAscendingList = []
            busDT.columnList = []

            busDT.save()

        except:
            self.logger.exception("")
            raise

    def test_Load(self):
        try:
            busDT = BUSDataTable()
            busDT.name = 'New'
            busDT.load()
            self.logger.debug(str(busDT))
        except:
            self.logger.exception("")
            raise

    def tearDown(self):
        pass

