"""
Created on Tue Feb 23 12:01:21 2016

@author: Ouranos
    > Used for extracing the data ... Will change this into mine 

    > Additions using gradient boosting regression technique gbr 

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.ensemble import ExtraTreesRegressor as algo
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import VarianceThreshold

from pyfm import pylibfm 

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
X_train , X_valid , y_train , y_valid  = train_test_split(train , labels , test_size = 0.10 , random_state = 0 )

print("Creating model")
est = pylibfm.FM(num_factors=50, num_iter=10, verbose=True, task="classification", initial_learning_rate=0.0001, learning_rate_schedule="optimal")


print("Fitting data")
est.fit(X_train,y_train)
print("Completed Fitting Data to Model")

print("Model Info")
print(" Important Features : ") 
print(est.feature_importances_)

print(" Validate Model ")
pred_valid_y = est.predict(X_valid)

mse = mean_squared_error(y_valid,pred_valid_y)
print("MSE : %.4f" % mse)

roc = roc_auc_score(y_valid,pred_valid_y)
print("ROC : %.4f" % roc)

print("Applying Model on Test Set")
preds = est.predict(test)
print(preds)
print("Obtained Prediction")

print("Creating csv file")
submission = pd.read_csv('sample_submission.csv')
submission["PredictedProb"] = preds
submission.to_csv('submission.csv', index=False)


print("Saving the Model")
from sklearn.externals import joblib
joblib.dump(est, './model/FactMach__norm_fea_ext_300.pkl')

# To load the model 
#est = joblib.load('extraTree.pkl')

