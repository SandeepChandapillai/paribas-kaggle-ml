"""
Created on Tue Feb 23 12:01:21 2016

@author: Ouranos
    > Used for extracing the data ... Will change this into mine 


    > Additions using gradient boosting regression technique gbr 

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor as algo 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score 
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


#Remove sparse columns
#print("Removing Spare Columns")
#cls = train.sum(axis=0)
#train = train.drop(train.columns[cls<10], axis=1)
#test = test.drop(test.columns[cls<10], axis=1)


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
print("Finding Best Parameters")

param_grid_grb = { 
        'loss': ['huber'] , 
        'n_estimators' : [100 , 500 , 1000 , 5000] , 
        'max_depth': [3 , 10 , 40 , 100] , 
        'max_features' : ["auto"] , 
        'min_samples_leaf' : [1 , 4 , 10 , 100] 
        }

est = GridSearchCV(estimator = algo() , param_grid = param_grid_grb , n_jobs = -1 ) 

#est = GradientBoostingRegressor( loss = 'huber' , n_estimators=1000, max_depth=10 , max_features = "auto" ,  min_samples_leaf=1 )
print("Model Info")

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

acc = accuracy_score(y_valid , pred_valid_y) 
print("ACC : %.4f" % acc)

score = est.score
print("Score : %.4f" % score)



#print(" Cross Validate Model ")
#scores = cross_val_score(est , train , labels , cv = 10 )  # Do cross validation 10 times 
#print(scores)

print("Applying Model on Test Set")
preds = est.predict(test)
print(preds)
print("Obtained Prediction")

print("Creating csv file")
submission = pd.read_csv('sample_submission.csv')
submission["PredictedProb"] = preds
submission.to_csv('submission.csv', index=False)


