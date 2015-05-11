'''
Created on May 6, 2015

@author: user
'''
import pandas as pd
import numpy as np

fname = "./../dataset/vor.csv"
paths = "./../dataset/"
filename = ['niederschlag.csv','aussen_temp.csv','ruck.csv','vor.csv','relative_humidity.csv','energyconsumption_resampled.csv']
class CReader:
    
    def __init__(self,filename=None):
        
        if(filename is None):
            raise Exception("File name not given")
        
        self.filename = filename
        
    def strip(self,text):
        try:
            return text.strip()
        except AttributeError:
            return text  
        
        
    def read(self):
        dataframe =  pd.read_csv(self.filename, parse_dates=[0], delimiter=";",index_col=0);
        print dataframe.head()
              
    def clean(self):
        dateparse = lambda x: pd.datetime.strptime(x, '%Y%m%d%H')
         
        dataframe =  pd.read_csv(self.filename, parse_dates=[1],date_parser=dateparse, delimiter=";",index_col=1,converters={u' MESS_DATUM':self.strip, 'NIEDERSCHLAGSHOEHE':self.strip});
         
        df =  dataframe[[u'NIEDERSCHLAGSHOEHE']].loc[ '20140606':'20140630']
       
        df.to_csv('niederschlag2.csv',sep=';', date_format='%d/%m/%Y %H:%M:%S' )
    
    def cleanenergy(self):
        figsz = (18,16)
        import matplotlib.pyplot as plt
        dataframe =  pd.read_csv(self.filename, parse_dates=[0], dayfirst = True,delimiter=";",index_col=0);
        dataframe.fillna(inplace = True,method="ffill")
        dataframe.fillna(inplace = True,method="bfill")
        dataframe = dataframe.resample(rule='H' )
        print dataframe.head()
        dataframe.fillna(inplace = True,method="ffill")
        dataframe.fillna(inplace = True,method="bfill")
        print np.where(pd.isnull(dataframe))
        print dataframe.head()
        dataframe.plot()
        plt.figsize = figsz
        plt.savefig("first.png")
        
        '''
        dfshifted =  dataframe- dataframe.shift()
        dfshifted.fillna(inplace = True,method="ffill")
        dfshifted.fillna(inplace = True,method="bfill")
        dfshifted.plot()
        plt.figsize = figsz
        plt.savefig("second.png")
        print dataframe.head(100),dfshifted.head(100) 
        
        dfshifted.to_csv(paths+'ruck_resampled.csv',sep=';', date_format='%d/%m/%Y %H:%M:%S' )    
        ''' 
        dataframe.to_csv(paths+'vor_resampled3.csv',sep=';', date_format='%d/%m/%Y %H:%M:%S' )
        
        
        
if __name__ == "__main__": 
    obj = CReader(fname)
    obj.cleanenergy()
