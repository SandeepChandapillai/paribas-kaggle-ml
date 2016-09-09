"""
    > Used for extracing the data ... Will change this into mine 
    > Additions using gradient boosting regression technique gbr 

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor as algo
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score 

import xgboost as xgb


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

print("Cleaning and formatting data")

train = np.asarray(train, dtype=np.float32)        
labels = labels.ravel()

model_name =  "xgboost_150_500_10"

est = xgb.Booster({'nthread':8}) #init model
est.load_model("model/" + model_name )

print("Saving the response from train set")
dx_train = xgb.DMatrix(train) 
y_pred = est.predict(dx_train)
pd.DataFrame({"ID": trainId, "PredictedProb": y_pred }).to_csv('./ver4/train_ds_0_' + model_name + '.csv' , index = False)

print("Saving the response from test set")
dx_test = xgb.DMatrix(test) 
y_pred = est.predict(dx_test)
pd.DataFrame({"ID": testId, "PredictedProb": y_pred }).to_csv('./ver4/test_ds_0_' + model_name + '.csv', index=False)


