"""
Created on Tue Feb 23 12:01:21 2016

@author: Ouranos
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
from sklearn.grid_search import GridSearchCV

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


print("Creating model")

# XGBoost params:
xgboost_params = { 
   "objective": "binary:logistic",
   "booster": "gbtree",
   "eval_metric": "auc",
   "eta": 0.01, # 0.06, #0.01,
   #"min_child_weight": 240,
   "subsample": 0.75,
   "colsample_bytree": 0.68,
   "max_depth": 15
}

#xgtrain = xgb.DMatrix(X_train , y_train)
#boost_round = 100 
#est = algo(n_estimators=300 , oob_score = True , random_state = 4 , n_jobs = -1 , max_features = "auto" , min_samples_leaf = 5)

param = {'max_depth' : (300 , 400 ) ,
        'n_estimators' : (250 , 300 , 400),
        'learning_rate' : (0.05 , 0.1 ),
        }


print("Fitting data")
#clf = xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round,verbose_eval=True,maximize=False)

est = xgb.XGBClassifier(max_depth=200 , n_estimators = 200 , learning_rate = 0.05)

est = GridSearchCV( est , param , n_jobs = -1)

#Make predict
print('Predict...')

est.fit(X_train,y_train)
print("Completed Fitting Data to Model")

print("Model Info")
#print(" Important Features : ") 

bst_param , score , _ = max(est.grid_scores_ , key=lambda x: x[1])

for param_name in sorted(param.key()):
    print( " %s : %r" % (param_name , bst_param[param_name]))



#print(est.feature_importances_)

print(" Validate Model ")

pred_valid_y = est.predict(X_valid)

mse = mean_squared_error(y_valid,pred_valid_y)
print("MSE : %.4f" % mse)

roc = roc_auc_score(y_valid,pred_valid_y)
print("ROC : %.4f" % roc)

print("Applying Model on Test Set")
#preds = est.predict(test)

#preds = clf.predict(xgTest, ntree_limit=clf.best_iteration)

preds = est.predict(test)

print(preds)
print("Obtained Prediction")

print("Creating csv file")
submission = pd.read_csv('sample_submission.csv')
submission["PredictedProb"] = preds
submission.to_csv('submission.csv', index=False)


print("Saving the Model")
from sklearn.externals import joblib
joblib.dump(est, './model/xgboost_no_fea_100')

