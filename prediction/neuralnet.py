'''
Created on May 12, 2015

@author: user
'''


import pickle
 
from pybrain.tools.shortcuts        import buildNetwork
from pybrain.supervised.trainers    import BackpropTrainer
from pybrain.datasets.supervised    import SupervisedDataSet

from pybrain.structure import FeedForwardNetwork, FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

paths = "./../dataset/"
inputparams = [u'Niederschlag', u'Aussentemperatur', u'Relative_Feuchte',u'hour']


inputparams = [u'Aussentemperatur', u'Relative_Feuchte',u'hour']
outputparams = [u'heatEnergy']

sub = [u'Niederschlag', u'Aussentemperatur', u'Relative_Feuchte',u'heatEnergy']


class CNeuralNet:
    def __init__(self,learningrate = 0.001,inputneurons = 2,hiddenneurons =50,outputneurons = 2,testondata= True, \
                 momentum = 0.2,train_percent = 99,recurnet = False):
        """
        Neural networks class
        assign a learning rate of your choice , default is 0.01
        inputneurons = number of neurons on input layer: can be set to the input dimension of the data
        hiddenneurons= keep it more than inputneurons in general
        outputneurons = output dimension of your data
        testondata = If you want to print out the performance of your neural net, defaults to true
        """
        assert (hiddenneurons > inputneurons), "Number of hiddenneurons can't be lesser than inputneurons"
        
        self.learningrate = learningrate
        self.inputneurons = inputneurons
        self.hiddenneurons = hiddenneurons
        self.outputneurons = outputneurons

        #momentum is the parameter to realize how efficiently the learning will get out of a local minima, 
        #not sure what to put as an appropriate value
        self.momentum = momentum

        #Construct network here
        self.mlpnetwork  = buildNetwork(self.inputneurons, self.hiddenneurons, self.outputneurons, bias=True,recurrent=recurnet)
        self.mlpnetwork.sortModules()
        print self.mlpnetwork,"-----------------------------------"
        self.trainer = None
        self.validation = testondata
        self.data = None
        
        self.learnedNetwork = None
        self.train_percent = (1. *train_percent )/100
        
        #creating a feedforward network
        
        
        
        
        self.ffn = FeedForwardNetwork()
        inlayer = LinearLayer(inputneurons)
        hiddenlayer1 = SigmoidLayer(4)
        hiddenlayer2 = SigmoidLayer(2)
        outlayer = LinearLayer(outputneurons)
        
        #assigning them to layers
        self.ffn.addInputModule(inlayer)
        self.ffn.addModule(hiddenlayer1)
        self.ffn.addModule(hiddenlayer2)
        self.ffn.addOutputModule(outlayer)
        
        #defining connections
        
        in_to_hidden1 = FullConnection(inlayer,hiddenlayer1)
        hidden1_to_hidden2 = FullConnection(hiddenlayer1,hiddenlayer2)
        hidden2_to_out = FullConnection(hiddenlayer2,outlayer)
        
        #explicitly adding them to network
        self.ffn.addConnection(in_to_hidden1)
        self.ffn.addConnection(hidden1_to_hidden2)
        self.ffn.addConnection(hidden2_to_out)
        
        #explicitly call sortmodules
        self.ffn.sortModules()
        
        print "created network successfully...."
        
        
    
    
    def train(self,filename,trainepochs = 1000):
        """
        train: call this function to train the network
        inputdata = set of input params
        trainepochs = number of times to iterate through this dataset
        """
        
        self.trainer = BackpropTrainer(self.mlpnetwork, dataset=self.data,verbose=True, learningrate=self.learningrate, momentum=self.momentum)
        
        #self.trainer = BackpropTrainer(self.ffn, dataset=self.data, verbose=True, learningrate=self.learningrate, momentum=self.momentum)
     
        #self.trainer.trainEpochs(epochs=trainepochs)
        print "training in progress..."
        
        
        for i in xrange( trainepochs ):
            mse = self.trainer.train()
            rmse = np.sqrt(mse)
            print "training RMSE, epoch {}: {}".format( i + 1, rmse )
        
        
        '''
        fl = filename.split('.log')[0] +str(time.strftime('%H_%M_%S'))+ "_Vivek_ActiveState_New.pickle"
        with open(fl, "wb") as f:
            pickle.dump(self.mlpnetwork, f)
        
        
        
        err = filename.split('.log')[0] + "_validation_errors.pkl"
        with open(err, "wb") as f:
            pickle.dump(self.trainer.validationErrors, f)
        '''
            
        print "training done..."
    
    def loadTrainedModel(self, pickleFile=None):
        """
        call this function to load the trained model  
        Please call loadTrainedModel once, before calling predict, so as to load the trained model and then predict things :)       
        """

        if pickleFile == None:
            # If there are many pre-computed neural-nets, load the first one
            from glob import glob
            pickleFile = glob("./*.pickle")[0]

        assert '.pickle' in pickleFile, "Invalid Neural-Net loaded..."
        with open(pickleFile, "rb") as f:
            self.mlpnetwork = pickle.load(f)
        
        return self.mlpnetwork
            
            
    def predict(self, testData):
        """
        testData = input data which is to be predicted on a given trained model
        if you trained the model earlier and want to reuse it
        """

        #assert (self.trainer != None) , "Train the model before you predict with it..."                    
        return self.mlpnetwork.activate(testData)
    
    
    def createTrainingData(self,filename,inputdim, outputdim):
        """
        create training data by reading file=filename
        inputdim = inputdimension of data
        outputdim = output dim expected
        """
        
        if filename is not None:
            finaldf = pd.read_csv(paths+filename, parse_dates=[0], delimiter=";",index_col=0);
            finaldf = finaldf.reset_index()     
            finaldf['hour'] = pd.DatetimeIndex(finaldf['TIMESTAMP']).hour 
            
            for col in finaldf:
                if(col not in ['TIMESTAMP','hour']):
                    print col
                    print "hhhhhhhhhhhhhhhhhhh"
                    finaldf[col] /= finaldf[col].iloc[0].astype(np.float64)
                
            print finaldf.head(10)          
            #split data into percentages
            msk = np.random.rand(len(finaldf)) < self.train_percent
            train = finaldf[msk].copy()
            test = finaldf[~msk].copy()
                      
            test = test.reset_index()
            train = train.reset_index()
            
            self.train_input =  train[inputparams]
            self.train_output = train[outputparams]
            
            #normalize train_output
            #self.train_output = 1. * self.train_output/self.train_output.max()
            #print self.train_output.head(10)
            
            
            self.test_input =  test[inputparams]
              
            self.test_output = test[outputparams] 
        
        self.data = SupervisedDataSet(inputdim,outputdim)
     
        totalLength = len(self.train_input)
        for line in xrange(0,totalLength-1):
            #print self.train_input.values[line], self.train_output.values[:,0][line]
            self.data.addSample(self.train_input.values[line], self.train_output.values[:,0][line])
        
        print "data loaded..."
        
        
    def createXORData(self,inputdim,outputdim):
 
        self.data = SupervisedDataSet(inputdim,outputdim)
        self.data.addSample([1,1],[0])
        self.data.addSample([1,0],[1])
        self.data.addSample([0,1],[1])
        self.data.addSample([0,0],[0])
            
        
            
if __name__ == "__main__":
    inputdim = len(inputparams)
    outputdim = len(outputparams)
    #inputdim = 2
    #outputdim = 1
    obj = CNeuralNet(inputneurons=inputdim,outputneurons=outputdim )
    obj.createTrainingData("final.csv",inputdim,outputdim)
    #obj.createXORData(inputdim,outputdim)
    obj.train("output", trainepochs=10000)
    
    '''
    print  (obj.mlpnetwork.activate([1,1]))
    print  (obj.mlpnetwork.activate([0,0]))
    print  (obj.mlpnetwork.activate([0,1]))
    print  (obj.mlpnetwork.activate([1,0]))
    '''