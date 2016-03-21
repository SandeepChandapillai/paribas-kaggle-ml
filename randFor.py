"""
    > Used for extracing the data ... Will change this into mine 


    > Additions using random forest 
    > convert the strings into numbers instead of dummies     
    > Learn to Validate the model 
"""


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score 
from sklearn.cross_validation import train_test_split 
   
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
#print(train.dtypes)

print("Cleaning and formatting data")
train = np.asarray(train, dtype=np.float32      
labels = labels.ravel()


print("Creating training set and validation set : 90 / 10 ")
X_train , X_valid , y_train , y_valid  = train_test_split(train , labels , test_size = 0.70 , random_state = 0 )


# Convert the data frame column into a single dim array 

print("Creating model")
#est = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.1,
#                                          loss='huber', min_samples_leaf=1, 
#                                          random_state=0)

#est = GradientBoostingClassifier(n_estimators=500, max_depth=5 , min_samples_leaf=3)
est = RandomForestRegressor(n_estimators=100 , oob_score = True , random_state = 42)

print("Fitting data")
est.fit(train,labels)

print("Completed Fitting Data to Model")
print("Model Info")
print(est.oob_score_)

y_oob = est.oob_prediction_
print "roc :" , roc_auc_score(labels , y_oob)

preds = est.predict(test)
print(preds[:,1])

submission = pd.read_csv('sample_submission.csv')
submission["PredictedProb"] = preds[:,1]
submission.to_csv('submission.csv', index=False)

fea_imp = pd.Series(est.feature_importances_ , index = train.columns)
fea_imp.sort()
fea_imp.plot(kind="barh", figsize=(7.6))


