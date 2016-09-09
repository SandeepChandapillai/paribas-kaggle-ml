"""
Created on Tue Feb 23 12:01:21 2016

    > Used for extracing the data ... Will change this into mine 

    > Additions using gradient boosting regression technique gbr 
    
    > Trying combining multiple models 

    > load already built models and combine them ... 


"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor as algo
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import mean_squared_error

# For Variance Selection 
from sklearn.feature_selection import VarianceThreshold

# For Additional Feature Selection 
from sklearn.svm import  LinearSVC
from sklearn.feature_selection import SelectFromModel 

# Adding more complexity to the model 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


from sklearn.preprocessing import OneHotEncoder

# To select the Model 
from sklearn.externals import joblib
# For file handling 
import os 


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


def testEstimator(est , y_valid , pred_valid_y):
    mse = mean_squared_error(y_valid,pred_valid_y)
    print("MSE : %.4f" % mse)

    roc = roc_auc_score(y_valid,pred_valid_y)
    print("ROC : %.4f" % roc)


def create_submission(est , pred , name):
    print("MODEL APPLIED ON TEST SET : ")
    print(name)
    print(preds)
    print("Obtained Prediction")
    
    name = 'submission_'+ str(name) + '.csv' 

    print("Creating csv file")
    submission = pd.read_csv('sample_submission.csv')
    submission["PredictedProb"] = preds
    submission.to_csv(name, index=False)




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
fea_sel = VarianceThreshold(threshold = 0.6)
train = fea_sel.fit_transform(train)
test = fea_sel.fit_transform(test)

print("After Removing Low Variance Features")
print(train.shape)
print(test.shape)

print("Selecting the best Features using Linear SVM ")
model_name_svc = 'model/linear_svc_pre_l1.pkl'
model_exist_svc = False 
if os.path.isfile(model_name_svc): 
    print('Model Exist')
    lsvc = joblib.load(model_name_svc)
    model_exist_svc = True 
else:
    lsvc = LinearSVC(C=0.1 , penalty = "l1" , dual = False).fit(train , labels)

if not (model_exist_svc):
    print("Saving the Model")
    joblib.dump(lsvc , model_name_svc)

model = SelectFromModel(lsvc , prefit = True)
train = model.transform(train)
test = model.transform(test)

print("After Selecting Important  Features")
print(train.shape)
print(test.shape)

print("Cleaning and formatting data")
train = np.asarray(train, dtype=np.float32)        
labels = labels.ravel()

# Convert the data frame column into a single dim array 
print("Creating training set and validation set : 90 / 10 ")
X_train , X_valid , y_train , y_valid  = train_test_split(train , labels , test_size = 0.10 , random_state = 0 )

print("Creating model")
model_name ='model/extraTree_fea_redu_900_60.pkl' 
model_exist = False
if os.path.isfile(model_name): 
    est = joblib.load(model_name)
    model_exist = True 
    print('Model Exist')
else:
    est = algo( n_jobs = -1  , n_estimators=300, max_features = "auto" ,  min_samples_leaf=1 )

print("Creating Complex Model")

#Using the Logistic Regression on top of Extra Tree Classfier 

est_ehc = OneHotEncoder()
est_lg = LogisticRegression()
est_ehc.fit(est.apply(X_train))
#Store the model 
joblib.dump(est_ehc , './model/est_ehc.pkl')

est_lg.fit(est_ehc.transform(est.apply(X_train)) , y_train)

joblib.dump(est_lg , './model/est_lg.pkl')

# Use Extra Tree Classifier Tranined on the entire set to determine the best features and use them


if not (model_exist):
    print("Fitting data")
    est.fit(X_train,y_train)
    print("Saving the Model")
    joblib.dump(est, model_name)

print("Completed Fitting Data to Model")
print("Model Info")
print(" Important Features : ") 
print(est.feature_importances_)

print(" Validate Model ")

print("Extra Classifer Model + LINEAR SVC")
pred_valid_y = est.predict(X_valid)
testEstimator(est , y_valid , pred_valid_y)
preds = est.predict(test)
create_submission(est , preds , 'extra_tree_svc')

print("Classifer Model + LINEAR SVC + Log Reg ")
pred_valid_y = est_lg.predict(est_ehc.transform(est.apply(X_valid)))
testEstimator(est_lg , y_valid , pred_valid_y)
preds = est_lg.predict(est_ehc.transform(est.apply(test)))
create_submission(est_lg , preds, 'extra_tree_svc_log')

# To load the model 
#est = joblib.load('extraTree.pkl')

