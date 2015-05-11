'''
Created on Apr 29, 2015

@author: user
'''

from utils import readdata as rd

import pandas as pd
pd.options.display.mpl_style = 'default'
from ggplot import *
import matplotlib.pyplot as plt
import random


class CMain:
    """
    Entry point for the entire program
    """
    def __init__(self):
        pass
    

def test():
    from ggplot import meat
    meat_lng = pd.melt(meat, id_vars=['date'])
    print meat[0:10]
    print meat_lng.head(5)
    
    print ggplot(aes(x='date', y='value', colour='variable'), data=meat_lng) + geom_line()    
 
def test2():
     
    #import pandas_datareader.data as web
    #import pandas.io.data as web
    
    #from pandas.io import data, wb # becomes
    from pandas_datareader import data
    
    all_data = {}
    for ticker in ['AAPL',  'IBM', 'YHOO', 'MSFT']:
        all_data[ticker] = data.get_data_yahoo(ticker, '1/1/2010', '1/1/2012')
    
 
    
    price = pd.DataFrame({tic: data['Adj Close'] for tic, data in all_data.iteritems()})
    import vincent 
    line = vincent.Line(price)
    line.axis_titles(x='Date', y='Price')
    line.legend(title='IBM vs AAPL')
    js = line.to_json('out.json',html_out=True,)
    
    
if __name__ == "__main__":
    #test()
    #test2()
     
    
    title = "Ruecklauftemperatur vs TIMESTAMP"
    xlabel = "TIMESTAMP"
    ylabel = "Ruecklauftemperatur"
    grid = True
    filename = "dataset/data_rueckleufttemp.csv"
    readobj = rd.CReadData(filename)
    df_ruecklauft = readobj.readfile()
 
    filename = "dataset/data_vorleufttemp.csv"
    readobj = rd.CReadData(filename)
    df_vorleufttemp = readobj.readfile()
 
    #print df_ruecklauft.head(10)
    #df_ruecklauft.set_index(keys='TIMESTAMP')
 
    #df = pd.DataFrame(({'timestamp':df_ruecklauft['TIMESTAMP'],'ruck':df_ruecklauft['Ruecklauftemperatur'], 'vor':df_vorleufttemp['Vorlauftemperatur'] }),index=)
   
    df = df_ruecklauft.join(df_vorleufttemp)
    df.plot()
    plt.savefig("df.png")
     
    print df[0:5]
     
    df1 = df.resample(rule='60Min',how=['mean']).fillna(method="ffill")
    print df1[0:5]
    df1.plot()
    plt.savefig("df1.png")
     
    
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
     
    '''
    import vincent 
    line = vincent.Line(df)
    line.axis_titles(x='Date', y='Value')
    line.legend(title='IBM vs AAPL')
    js = line.to_json('out.json',html_out=True,)
    '''
    '''
    #outdf = pd.melt(df, id_vars=['timestamp']).dropna()

    #plt.figure()
    #plt.plot_date(x=df['timestamp'], y= df['ruck'],fmt="r-")
    #plt.plot_date(x=df['timestamp'], y= df['vor'],fmt="b-")
    
    #plt.show()
    
    #print ggplot( aes(x='timestamp', y='ruck'),data=df) + geom_line(color='lightblue') + stat_smooth(span=.15)
    #plt.show(1)
    #print ggplot(df, aes(x='timestamp', y='vor')) + geom_line() + geom_line(df, aes(x='timestamp', y='ruck'))
    print ggplot(outdf, aes('timestamp','value',color='variable')) \
    + geom_line(size=2.) \
    + geom_hline(yintercept=0, color='black', size=1.7, linetype='-.') \
    + scale_x_date(labels='%b %d %y',breaks=date_breaks('months') ) \
    + theme_seaborn(style='whitegrid') \
    + ggtitle('Compare temp vs ruecklaufts') 
    
    plt.show(1)
    '''
    '''
    retPlot_YTD = ggplot(outdf, aes('timestamp','value',color='variable')) \
    + geom_line(size=2.) \
    + geom_hline(yintercept=0, color='black', size=1.7, linetype='-.') \
    + scale_y_continuous(labels='percent') \
    + scale_x_date(labels='%b %d %y',breaks=date_breaks('months') ) \
    + theme_seaborn(style='whitegrid') \
    + ggtitle('Compare temp vs ruecklaufts') 

    fig = retPlot_YTD.draw()
    ax = fig.axes[0]
    offbox = ax.artists[0]
    offbox.set_bbox_to_anchor((1, 0.5), ax.transAxes)
    fig.show()
    '''
    
    
    #print ggplot(aes(x='timestamp', y='value', colour='variable'), data=outdf) +  stat_smooth(span=.15)
    
    '''
    print df.head(9)
    print ggplot(aes(x='timestamp', y='ruck'), data=df) + \
    geom_line(color='lightblue') + \
    geom_line(aes(x='timestamp', y='vor'), data=df,color='red') + \
    stat_smooth(span=.15) + \
    ggtitle("Compare temp vs ruecklaufts") 
    
    #xlab("Date") + \
    #ylab("Ruecklauftemperatur")
        
    plt.figure()
    #plt.plot_date(x=df_ruecklauft['TIMESTAMP'], y=df_ruecklauft['Ruecklauftemperatur'],fmt='r-')
    df_ruecklauft.plot()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(grid)
    
    
    '''