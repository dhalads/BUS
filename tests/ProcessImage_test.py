import unittest
import logging
import json, logging.config
import sys
sys.path.append('Segmentation')
sys.path.append('tests')
from BUSSegmentor import BUSSegmentor
from BUSSegmentorList import BUSSegmentorList
from BUSDataTable import BUSDataTable
from BUSDataTable import BUSDataTable
from ProcessImage import ProcessImage
from ProcessImage import busUI
from ProcessImage import singleUI

class ProcessImage_test(unittest.TestCase):

    logger = logging.getLogger("BUS." + __name__)
    
    def setUp(self):
        with open('logging-config.json', 'rt') as f:
            config = json.load(f)
            logging.config.dictConfig(config)

    def test_sourceList(self):
        try:
            single = singleUI()
            sourceList = single.getSourceList()
            self.logger.debug(str(sourceList))
        except:
            self.logger.exception("")
            raise

    def tearDown(self):
        pass

