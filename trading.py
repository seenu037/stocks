# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 22:23:21 2019

@author: Haritha
"""
import pandas as pd
import plotly
from plotly import graph_objs as go
import numpy as np
#month=2018
res=pd.DataFrame()
df = pd.read_csv('NIFTY_F1_2019-1F.csv')
df['date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y %H:%M")
df.columns=['Date','op','high','low','close','volume','date']
#df = df[df['date'].map(lambda x: x.month) == month]
year=df['date'].dt.year.max()
df['avg_price']=df[['op','high','low','close']].mean(axis=1)
df['avg_price_shift']=df['avg_price'].shift(1)
df['price_delta']=df['avg_price']-df['avg_price_shift']
df=df.dropna(subset=['price_delta']).reset_index(drop=True)
for threshold in list(np.arange(20,150,10)):
    temp=pd.DataFrame()
    sum=0
    n=0
    df['cum_sum']=0
    for i in range(len(df.index)):
        sum=sum+df['price_delta'][i]
        df['cum_sum']['i']=sum
        if (sum>threshold or sum<-threshold):
            dt=df['date'][i]
            op=df['op'][n]
            high=df.iloc[n:i+1,:]['high'].max()
            low=df.iloc[n:i+1,:]['low'].min()
            cl=df['close'][i]
            p=sum
            sum=0
            temp1=pd.DataFrame([dt,op,high,low,cl,p])
            temp=temp.append(temp1.T)
            n=i+1
            temp.columns=['date','op','high','low','close','cum_sum']
            temp['p_1']=np.where(temp['cum_sum']<0,-1,1)
            temp.sort_values(by=['date'],inplace=True)
            temp['month']=temp['date'].dt.month
            temp['num'] = (temp.p_1 != temp.p_1.shift()).cumsum()
            
            temp['diff']=temp.groupby(['p_1','num']).cumcount()+1
            #temp.groupby(['p_l','num'],as_index=False)['diff'].max()
            temp3=temp.groupby(['p_1','num'],as_index=False)['diff'].max()
            temp3=temp.rename(columns={'diff':'diff_max'})
            temp=pd.merge(temp,temp3,how='left',on=['p_1','num'])
            temp=temp[temp['diff_max']>1]
            temp['num'] = (temp.p_1 != temp.p_1.shift()).cumsum()
            temp['diff']=temp.groupby(['p_1','num']).cumcount()+1
            
            temp['date']=pd.to_datetime(temp['date'])
            temp3=temp[temp['diff']==2]
            temp3['B_S']=np.where(temp3['p_1']==-1,'S','B')
            temp3['B_S2']=temp3['B_S'].shift(-1)
            temp3['close2']=temp3['close'].shift(-1)
            #temp3['close3']=temp3['close'].shift(-1)
            #temp3=temp3.dropna(subset=['B_S2']).reset_index(drop=True)
            temp3[['B_S','B_S2']].groupby(['B_S','B_S2'])['B_S'].count()
            #temp3['profit']=temp3['close']-temp3['close2']
            temp3['profit']=np.where(((temp3['B_S']=='S') & (temp3['B_S2']=='B')),temp3['close']-temp3['close2'],temp3['close2']-temp3['close'])
            temp3[['month','profit']].groupby(['month'])['profit'].sum()
            a1=temp3['profit'].sum()-len(temp3)*4
            a2=temp3[temp3['profit']<0]['profit'].sum()
            a3=temp3[temp3['profit']>0]['profit'].sum()
            a4=len(temp3)
            res1=pd.DataFrame([threshold,a1,a2,a3,a4])
            res=res.append(res1.T)
res.columns=['thresholds','index_pts','loss','profit','#trades']
#res2=res[['thresholds','index_pts','num_trades']]
res2=res.set_index(['thresholds'])
import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='Price in $')
res2.plot(ax=ax1, lw=2.)
plt.grid()
plt.savefig(str(year)+'_Intraday'+'.png')


#signals['positions'] = signals['signal'].diff()
fig = go.Figure(data=[go.Candlestick(x=temp['date'],
                open=temp.op,
                high=temp.high,
                low=temp.low,
                close=temp.close)])
#plotly.offline.plot(fig)
trace = go.Candlestick(x=temp['date'],open=temp.op,high=temp.high,low=temp.low,close=temp.close)
layout = go.Layout(xaxis = dict({'type': 'category','showticklabels': False},rangeslider = dict(visible = True)),yaxis = dict(tickformat = '0'), width=1350,height= 640)
plotly.offline.plot(go.Figure(data=[trace],layout=layout), filename= (str(year)+'.html'))
#plotly.offline.plot(go.Figure(data=[trace]))
##########################MACD#######################
temp4=temp3[['date','p_1']]
df2=pd.merge(df,temp4,how='left')

df2=df2.set_index(['date'])

# Import 'pyplot' module as 'plt'
import matplotlib.pyplot as plt

# Initialize the plot figure
fig = plt.figure()

# Add a subplot and label for y-axis
ax1 = fig.add_subplot(111, ylabel='Price in $')

#plot the closing price
df2['avg_price'].plot(ax=ax1, color='r', lw=1.)
# Plot the buy signals
ax1.plot(df2.loc[df2.p_l==1].index,df2.avg_price[df2.p_l==1],
         
         '^', markersize=10, color='m')

# plot the sell signals
ax1.plot(df2.loc[df2.p_l==-1].index,df2.avg_price[df2.p_l==-1],
         
         'v', markersize=10, color='k')
# show the plot
#plt.show()

plt.savefig(str(year)+'_'+str(month)+'_'+str(threshold)+'.png')





####################################
# Initialize the short and long windows
short_window = 12
long_window = 26
month=9
# Initialize the 'signals' DataFrame with the 'signal' column
signals = pd.read_csv('NIFTY_F1.txt', sep=",", header=None, parse_dates=[[1,2]])
signals.columns=['date','nifty_f1','op','high','low','close','volume','turnover']
signals = signals[signals['date'].map(lambda x: x.month) == month]
signals=signals.set_index(['date'])
signals['signal'] = 0.0

# Create short simple moving average over the short window
signals['short_mavg'] = signals['close'].rolling(window=short_window, min_periods=1, center=False).mean()

# Create long simple moving average over the long window
signals['long_mavg'] = signals['close'].rolling(window=long_window, min_periods=1, center=False).mean()

# Create signals
signals['signal'][short_window:] = np.where(signals['short_mavg'][short_window:]
                                            > signals['long_mavg'][short_window:], 1.0, 0.0)

# Generate trading orders
signals['positions'] = signals['signal'].diff()

# Print 'signals'
print(signals)

# Import 'pyplot' module as 'plt'
import matplotlib.pyplot as plt

#Initialize the plot figure
fig = plt.figure()

#Add a subplot and label for y-axis
ax1 = fig.add_subplot(111, ylabel='Price in $')

# Plot the closing price
signals['close'].plot(ax=ax1, color='r', lw=1.)

# Plot the short and long moving averages
#signals[['short_mavg', 'long_mavg']].plot(ax=ax1, lw=2.)

# Plot the buy signals
ax1.plot(signals.loc[signals.positions == 1.0].index,
         signals.short_mavg[signals.positions == 1.0],
         '^', markersize=10, color='m')

# Plot the sell signals
ax1.plot(signals.loc[signals.positions == -1.0].index,
         signals.short_mavg[signals.positions == -1.0],
         'v', markersize=10, color='k')

# Show the plot
plt.show()
            
            
            
            
            