# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 23:40:00 2021

@author: evgeny
"""
#Project's aim: To allow localizaton of corantine measures due to early identification of covid spread
#The structure of analysis : 
#Getting data regarding number of patients by city 
#Normalizing data, fixing mistakes 
#Creating predictions for spreading rate per city 
#Using formla identified in the artickle x35
import io
import numpy as np
import numpy.ma as ma
import pandas as pd 
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics  import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model, metrics 
import requests

#checking the type of data and versions
print('{} version: {}' .format(np.__name__, np.__version__))
print('{} version: {}' .format(pd.__name__,   pd.__version__ ))
#getting the information from data 
url = "https://raw.githubusercontent.com/idandrd/israel-covid19-data/master/CityData.csv"
s=requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')))
#information regarding object data 
df.info()
##
df.describe()
#changing data from nan to 0 
df2=df.fillna(0)
#checking data with nuls
print(df2)
# going to janbiel city data 
print(df2.loc[201,:])

#getting the function 
#functioin named city : checks the regression variables
#function allows to calculate the coefficient variables
def city(x,y):
    regr= linear_model.LinearRegression()
    X = x.reshape(-1,1)
    Y = y.reshape(-1,1)
    X_train, X_test, y_test = train_test_split(X,Y, test_size=0.4, random_state=1)
    #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=1)
    regr.fit(X_train, y_train)
    #adding values into data array
    d=regr.fit(X_train, y_train)
    print('Regression Coefficients: {}'.format(regr.coef_[0]))
    y_pred =regr.predict(X_test)
    plt.scatter(X_test,y_test,  color='green')
    plt.plot(X_test, y_pred,color='blue', linewidth=1)
    regr.score(X,y)
    
# Getting x and y variables 
# There are two variabels the first are dates     
k=df2.keys()
k=np.array([])
G=(k[2:])
print('Y is',G)
#function 
for idx, x in np.ndenumerate(G):
    k=np.append(G,idx)
    print(k)

l=df2.loc[201]
t=np.array(l)
y=t[2:]
'printing the date two first data'
print('diffe',G)
#print('number of y', Y.count())
print("number of items",len(y))  
city(G,y)  
    

#'getting x and y'
Y=df2.loc[201,:]
'getting date'
U=Y.keys()
'array'-
R=np.array(U)
#data
G=(R[2:])
value=0
k=np.array([])
print(k)

for idx, x in np.ndenumerate(G):          
         k=np.append(k,idx)
         print(k)
         #k[idx==idx]      
print("number of x",G)
l=df2.loc[201]
t=np.array(l)
y=t[2:]
'printing the date two first data'
print('diffe', k)
print('number of y', Y.count())
print("number  of items",len(y))
city(k,y)
