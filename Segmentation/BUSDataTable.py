import json
from types import SimpleNamespace
import logging
import pandas as pd
from ipywidgets import widgets, interactive

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
        self.selectFollow = None

        self.df = None

    def __str__(self):
        return str(self.__dict__)

    def save(self):
        try:
            filename = 'data/' + self.name
            # https://www.geeksforgeeks.org/encoding-and-decoding-custom-objects-in-python-json/
            included_keys = ['source', 'queryString', 'sortList', 'sortAscendingList', 'columnList', 'isFollowImageList', 'isFollowSingle']
            # output_dict = {k:v for k,v in self.__dict__.items() if k in included_keys}
            json_object = json.dumps(self, indent = 4, default=lambda o: {k:v for k,v in o.__dict__.items() if k in included_keys})
            with open(filename, "w") as outfile:
                outfile.write(json_object)
        except:
            self.logger.exception("")
            raise

    def load(self):
        try:
            filename = 'data/' + self.name
            # https://stackoverflow.com/questions/6578986/how-to-convert-json-data-into-a-python-object
            with open(filename, 'r') as openfile:
                # Reading from json file
                json_object = json.load(openfile)
            self.source = json_object.get('source')
            self.queryString = json_object.get('queryString')
            self.sortList = json_object.get('sortList')
            self.sortAscendingList = json_object.get('sortAscendingList')
            self.columnList = json_object.get('columnList')
            self.isFollowImageList = False
            self.isFollowSingle = False
        except:
            self.logger.exception("")
            raise

    def init_df(self):
        error = False
        message = ''
        try:
            df = pd.read_csv(self.source, header=0)
            self.logger.debug(df.columns)
            if len(self.queryString) > 0:
                df.query(self.queryString, inplace=True)
            if len(self.sortList) > 0 :
                df.sort_values(by=self.sortList, inplace=True, ascending=self.sortAscendingList)
            if len(self.columnList) > 0:
                df = df[self.columnList]
        except Exception as e:
            self.logger.exception("")
            message = e
            error = True
        if error :
            output = (False, message)
        else:
            self.df = df
            message = 'Good load' 
            output = (True, message)
        return output

    def get_panel(self, id=None, idList=None):
        try:
            if self.selectFollow == 'ImageList' and idList is not None :
                queryString = 'id in [' + ",".join(map(str, idList)) + ']'
            elif self.selectFollow == 'Single' and id is not None :
                queryString = 'id in [' + str(id) + ']'
            else:
                queryString = None
            self.logger.debug("queryString=%s", queryString)
            if queryString is not None:
                new_df = self.df.query(queryString, inplace=False)
            else:
                new_df = self.df
            box_layout = widgets.Layout(overflow='scroll',
                    border='3px solid black',
                    width='100%',
                    max_height='300px',
                    # flex_flow='row nowrap',
                    display='flex')
            output = widgets.Output(layout=box_layout)
            output.append_display_data(new_df)
            return output
        except Exception as e:
            self.logger.exception('')
            raise

    def get_id_list(self):
        try:
            df = self.df
            output = self.df['id'].to_list()
        except Exception as e:
            self.logger.exception('')
            raise
        return output

