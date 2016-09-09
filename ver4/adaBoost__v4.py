"""
Created on Tue Feb 23 12:01:21 2016

    > Used for extracing the data ... Will change this into mine 

    > Additions using gradient boosting regression technique gbr 

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor as algo
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import VarianceThreshold

"""
    Input : string 
    out :   number encoding of stiring 
"""
def con_str_int(s):
    num = 0 ;  
    for i in s:
        num += ord(i)
    num = np.float32(num)
    return num

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# label is y 
labels = train.pop("target") 
trainId = train.pop("ID")
testId = test.pop("ID")

print ("Filling in missing values...")
train.fillna(-1 , inplace = True)
test.fillna(-1 , inplace = True)

print ("Converting Strings into integer columns ...")
train = train.applymap(lambda s: con_str_int(s) if isinstance(s , (str , unicode)) else s)
test = test.applymap(lambda s: con_str_int(s) if isinstance(s , (str , unicode)) else s)

print("Input Data set :")
print(train.describe())

#Remove Columns with low variance :
print("Removing Columns with Low Variance")
fea_sel = VarianceThreshold(threshold = 0.3)
train = fea_sel.fit_transform(train)
test = fea_sel.fit_transform(test)

print("Cleaning and formatting data")

train = np.asarray(train, dtype=np.float32)        
labels = labels.ravel()

# Convert the data frame column into a single dim array 
print("Creating training set and validation set : 90 / 10 ")
X_train , X_valid , y_train , y_valid  = train_test_split(train , labels , test_size = 0.20 , random_state = 0 )

from sklearn.externals import joblib
print("Opening model")
est = joblib.load('./model/adaBoost__no_fea_ext_400.pkl')

print(" Validate Model ")
pred_valid_y = est.predict(X_valid)

mse = mean_squared_error(y_valid,pred_valid_y)
print("MSE : %.4f" % mse)

roc = roc_auc_score(y_valid,pred_valid_y)
print("ROC : %.4f" % roc)

print("Saving the response from train set")
y_pred = est.predict(train)
pd.DataFrame({"ID": trainId, "PredictedProb": y_pred }).to_csv('./ver4/train_ada_ds_2_400.csv' , index = False)

print("Saving the response from test set")
y_pred = est.predict(test)
pd.DataFrame({"ID": testId, "PredictedProb": y_pred }).to_csv('./ver4/test_ada_ds_2_400.csv', index=False)


