'''
Created on Apr 29, 2015

@author: user
'''
import pandas as pd

class CReadData:
    def __init__(self,filename=None):
        if filename == None:
            raise Exception("Please give the filename")

        self.filename = filename
        self.dataframe = None
        
    def readfile(self,**kwargs):
        '''
        accepts keyworded arguments for pandas.read_csv
        '''
        self.dataframe = pd.read_csv(self.filename, parse_dates=[0], dayfirst=False,delimiter=";",index_col=0);
        print self.dataframe.columns
        if(self.dataframe is not None):
            return self.dataframe
        raise Exception("Dataframe is not initialized")

    
    