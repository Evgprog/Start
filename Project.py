# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 23:40:00 2021

@author: evgeny
"""
import io
import numpy as np
import numpy.ma as ma
import pandas as pd 
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics  import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split
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

#getting the fucntion 
#functioin named city : checks the regression variables






