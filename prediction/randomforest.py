'''
Created on May 7, 2015

@author: user
'''

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
 
paths = "./../dataset/"
filename = ['niederschlag.csv','aussen_temp.csv','ruck_resampled3.csv','vor_resampled3.csv','relative_humidity.csv','energyconsumption_resampled.csv']


class CRandomForest:
    def __init__(self,train_percent = 80):
        
        self.train_percent = 1. * train_percent/100 
        
        self.train_input = None
        self.train_output = None
        self.test_input = None
        self.test_output = None
        
    def prepare_data(self):
        df_comp = pd.DataFrame()

  
        for f in filename:
            tempdf = pd.read_csv(paths+f, parse_dates=[0], delimiter=";",index_col=0);
           

            if df_comp.empty:
                df_comp = tempdf.loc['20140616120000':].copy() 
                
            else:
             
                tf = tempdf['20140616120000':]
                #print "its not empty-------->",tf.head()
                df_comp = df_comp.join(tf)
                #print df_comp.columns, df_comp.head()
 
        df_comp.to_csv(paths+'final.csv',sep=';', date_format='%d/%m/%Y %H:%M:%S' )
        
    def read_and_predict_RandomForest(self,filename=None):
        if filename is not None:
            finaldf = pd.read_csv(paths+filename, parse_dates=[0], delimiter=";",index_col=0);
            
            msk = np.random.rand(len(finaldf)) < self.train_percent
            train = finaldf[msk].copy()
            test = finaldf[~msk].copy()
            
            train = train.reset_index()
          
            test = test.reset_index()
            self.train_input =  train[[u'Niederschlag', u'Aussentemperatur', u'Ruecklauftemperatur', u'Vorlauftemperatur', u'Relative_Feuchte']]
            self.train_output = train[[u'heatEnergy']]
            #print self.train_input.head(),self.train_output.head()
            
            self.test_input =  test[[u'Niederschlag', u'Aussentemperatur', u'Ruecklauftemperatur', u'Vorlauftemperatur', u'Relative_Feuchte']]
              
            self.test_output = test[[u'heatEnergy']]         
            #print self.test_input.head(),self.test_output.head()
            
            #check if the dataframe is null at any index
            #print np.where(pd.isnull(self.train_input)),np.any(pd.isnull(self.train_output))
            
            regressor = RandomForestRegressor(n_estimators=150)
            clf = regressor.fit(self.train_input,self.train_output)
            print clf.score(self.test_input,self.test_output)

            df = regressor.predict(self.test_input)

            #fig, axes = plt.subplots(nrows=2, ncols=2)
            comparedf = pd.DataFrame({'predicted':df}).join(self.test_output)
            print comparedf.head(10)
            print "r2 score--->" , r2_score(comparedf[u'heatEnergy'],comparedf[u'predicted'])
            print "mse is-------", np.mean((comparedf[u'heatEnergy']-comparedf[u'predicted'])**2)
            comparedf.plot()

            #df1.plot(ax=axes[0,0])
            #df2.plot(ax=axes[0,1])
            plt.show()
            
    def read_and_predict_SVMRegressor(self,filename=None):
        if filename is not None:
            finaldf = pd.read_csv(paths+filename, parse_dates=[0], delimiter=";",index_col=0);
            
            msk = np.random.rand(len(finaldf)) < self.train_percent
            train = finaldf[msk].copy()
            test = finaldf[~msk].copy()
            
            train = train.reset_index()
          
            test = test.reset_index()
            
            self.train_input =  train[[u'Niederschlag', u'Aussentemperatur', u'Ruecklauftemperatur', u'Vorlauftemperatur', u'Relative_Feuchte']]
            self.train_output = train[[u'heatEnergy']]
            
            #print self.train_input.head(),self.train_output.head()
            print self.train_input.values.shape ,"\n",self.train_output.values.shape
            print self.train_input.values[0:10] ,"\n",self.train_output.values[:,0] 
            
            self.test_input =  test[[u'Niederschlag', u'Aussentemperatur', u'Ruecklauftemperatur', u'Vorlauftemperatur', u'Relative_Feuchte']]
              
            self.test_output = test[[u'heatEnergy']]  
                
                
            '''    
            self.train_input = np.sort(5 * np.random.rand(40, 1), axis=0)
            self.train_output = np.sin(self.train_input).ravel()
            
            ###############################################################################
            # Add noise to targets
            self.train_output[::5] += 3 * (0.5 - np.random.rand(8))   
            print self.train_input ,"\n",self.train_output
            '''
            #check if the dataframe is null at any index
            #print np.where(pd.isnull(self.train_input)),np.any(pd.isnull(self.train_output))
        
            regressor = svm.SVR(kernel='rbf', C=1e3, gamma=0.1)
           
            clf = regressor.fit(self.train_input.as_matrix(),self.train_output.values[:,0] )
            
            
            print clf.score(self.test_input,self.test_output)

            df = regressor.predict(self.test_input)

            #fig, axes = plt.subplots(nrows=2, ncols=2)
            comparedf = pd.DataFrame({'predicted':df}).join(self.test_output)
            print comparedf.head(10)
            print "r2 score--->" , r2_score(comparedf[u'heatEnergy'],comparedf[u'predicted'])
            print "mse is-------", np.mean((comparedf[u'heatEnergy']-comparedf[u'predicted'])**2)
            comparedf.plot()

            #df1.plot(ax=axes[0,0])
            #df2.plot(ax=axes[0,1])
            plt.show()  
            
    

''' 
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print df.head()
 
train, test = df[df['is_train']==True], df[df['is_train']==False]

features = df.columns[:4]

clf = RandomForestClassifier(n_jobs=4)
y, _ = pd.factorize(train['species'])
print y,"--------------"
clf.fit(train[features], y)
 
preds = iris.target_names[clf.predict(test[features])]
print pd.crosstab(test['species'], preds, rownames=['actual'], colnames=['preds'])

'''           
            
if __name__ == "__main__":
    
    obj = CRandomForest()
    
    obj.read_and_predict_SVMRegressor("final.csv")
