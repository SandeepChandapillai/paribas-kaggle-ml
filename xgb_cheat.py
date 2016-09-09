"""
Created on Tue Feb 23 12:01:21 2016

    > Used for extracing the data ... Will change this into mine 

    > Additions using gradient boosting regression technique gbr 

    > Use Cross Validation 
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

# Convert the data frame column into a single dim array 
print("Creating training set and validation set : 90 / 10 ")
X_train , X_valid , y_train , y_valid  = train_test_split(train , labels , test_size = 0.10 , random_state = 0 )

dx_train = xgb.DMatrix(X_train , y_train)
#dx_valid = xgb.DMatrix(X_valid , y_valid)

print("Creating model")

#   # Run settings
#   md <- 11
#   ss <- 0.96
#   cs <- 0.45
#   mc <- 1
#   np <- 1
#
#   list(  objective           = "binary:logistic", 
#                   booster             = "gbtree",
#                   eval_metric         = "logloss",
#                   eta                 = 0.01,
#                   max_depth           = md,
#                   subsample           = ss,
#                   colsample_bytree    = cs,
#                   min_child_weight    = mc,
#                   num_parallel_tree   = np
#   )

#evalList = [(dx_valid,'eval'), (dx_train,'train')] 


# XGBoost params:
xgboost_params = { 
   "objective": "binary:logistic",
   "booster": "gbtree",
   "eval_metric": ["auc" , "logloss"],
   "eta": 0.01, # 0.06, #0.01,
   "min_child_weight": 10,
   "subsample": 0.96,
   "colsample_bytree": 0.45,
   "max_depth": 11
}
xgboost_params['nthread'] = 8 ; 

print("Fitting data")
#Make predict
print('Predict...')


#xgtrain = xgb.DMatrix(X_train , y_train)
boost_round = 100 
est = xgb.train(xgboost_params , dx_train , boost_round )

print("Completed Fitting Data to Model")


print("Model Info")
#print(" Important Features : ") 
#print(est.feature_importances_)

print(" Validate Model ")

dx_valid = xgb.DMatrix(X_valid)
pred_valid_y = est.predict(dx_valid)
mse = mean_squared_error(y_valid,pred_valid_y)
print("MSE : %.4f" % mse)

roc = roc_auc_score(y_valid,pred_valid_y)
print("ROC : %.4f" % roc)

print("Applying Model on Test Set")

dx_test = xgb.DMatrix(test)
preds = est.predict(dx_test)

print(preds)
print("Obtained Prediction")

print("Creating csv file")
submission = pd.read_csv('sample_submission.csv')
submission["PredictedProb"] = preds
submission.to_csv('submission.csv', index=False)

print("Saving the Model")
est.save_model("model/xgboost_100_11_10_noEval")


