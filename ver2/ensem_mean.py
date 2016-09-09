"""
Created on Tue Feb 23 12:01:21 2016

    > Used for extracing the data ... Will change this into mine 

    > Additions using gradient boosting regression technique gbr 
    
    > Trying combining multiple models 

    > load already built models and combine them ... 
    
    > Using a simple formula... Use multiple methods and then take the mean of the different predictions 
    
    > Each model for now is trained on entire date set : 
        > Fro better accurarcy may be train on different parts of the data set 

    > Models used : 
        RandomForest 
        Neural Network 
        ExtraTree 
        GBR 
        



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

# For exiting program on failure to read file 


from sys import exit
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


def create_submission( pred , name):
    print("MODEL APPLIED ON TEST SET : ")
    print(name)
    print(preds)
    print("Obtained Prediction")
    name = 'submission'+ str(name) + '.csv' 
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
train_red_var = fea_sel.fit_transform(train)
test_red_var = fea_sel.fit_transform(test)

#Remove Columns with low variance :
print("Removing Columns with Low Variance")
fea_sel = VarianceThreshold(threshold = 0.3)
train_red_var_1 = fea_sel.fit_transform(train)
test_red_var_1 = fea_sel.fit_transform(test)


print("After Removing Low Variance Features")
print(train_red_var.shape)
print(test_red_var.shape)

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
train_red = model.transform(train_red_var)
test_red = model.transform(test_red_var)
# Some models use the entire model Others use a reduced portion of the model 

print("After Selecting Important  Features")
print(train_red.shape)
print(test_red.shape)

print(" Original Data set")
print(train.shape)
print(test.shape)

print("Cleaning and formatting data")
train_red = np.asarray(train_red, dtype=np.float32)        
train = np.asarray(train, dtype=np.float32)        
labels = labels.ravel()

# Convert the data frame column into a single dim array 
print("Creating training set and validation set : 90 / 10 ")
X_train_red , X_valid_red , y_train_red , y_valid_red  = train_test_split(train_red , labels , test_size = 0.10 , random_state = 0 )

X_train , X_valid , y_train , y_valid  = train_test_split(train , labels , test_size = 0.10 , random_state = 0 )

X_train_var , X_valid_var , y_train_var , y_valid_var  = train_test_split(train_red_var_1 , labels , test_size = 0.10 , random_state = 0 )

# CLASSIFIER LIST # Holds classifiers which work on non reduced data set 
classifier_org = []
classifier_lowVar = []


print("Creating model")
model_name ='model/extraTree_fea_redu_900_60.pkl' 
model_exist = False
if os.path.isfile(model_name): 
    est = joblib.load(model_name)
    model_exist = True 
    print('Model Exist')
else:
    est = algo( n_jobs = -1  , n_estimators=300, max_features = "auto" ,  min_samples_leaf=1 )


if not (model_exist):
    print("Fitting data")
    est.fit(X_train_red,y_train_red)
    print("Saving the Model")
    joblib.dump(est, model_name)

print("Creating Complex Model")
#Using the Logistic Regression on top of Extra Tree Classfier 

#   est_ehc = OneHotEncoder()
#   est_lg = LogisticRegression()
#   est_ehc.fit(est.apply(X_train_red))
#   est_lg.fit(est_ehc.transform(est.apply(X_train_red)) , y_train_red)


# Use Extra Tree Classifier Tranined on the entire set to determine the best features and use them


# IMPORT Trained RANDOM FOREST MODEL ORIGINAL SET  MODEL # 3

model_name =  './model/random_For__fea_no_ext_300_2.pkl'
if os.path.isfile(model_name): 
    est_ranFor_300 = joblib.load(model_name)
    print(" RANDOM FOREST 300 LOADED ") 
else:
    print(" CANNOT LOAD 3") 
    exit(0)

classifier_org.append(est_ranFor_300);

# IMPORT TRAINED GBR LOW VAR  MODEL #4 

model_name =  './model/gradientBoost__fea_ext_100.pkl'
if os.path.isfile(model_name): 
    est_gbr_100 = joblib.load(model_name)
    print(" GBR 100 LOADED ") 
else:
    print(" CANNOT LOAD 4") 
    exit(0)

classifier_lowVar.append(est_gbr_100)

# IMPORT TRAINED ADABOOST LOW VAR  MODEL #5 
model_name =  './model/adaBoost__no_fea_ext_300.pkl' ; 
if os.path.isfile(model_name): 
    est_adaBoost = joblib.load(model_name)
    print(" ADA BOOST LOADED ") 
else:
    print(" CANNOT LOAD 5") 
    exit(0)

classifier_lowVar.append(est_adaBoost)

#   # IMPORT TRAINED NEURAL NETWORK  MODEL #6 
#   model_name = './model/neural_net.pkl' ; 
#   if os.path.isfile(model_name): 
#       est_neural_net = joblib.load(model_name)
#       print(" NEURAL NET  LOADED ") 
#   else:
#       print(" CANNOT LOAD 6") 
#       exit(0)
#
#   classifier.append(est_neural_net)

# IMPORT RAND FOREST WITH BAGGING ORIGINAL DATA  MODEL#7
model_name = './model/random_For__bagging' ; 
if os.path.isfile(model_name): 
    est_ranFor_bag = joblib.load(model_name)
    print(" RANDOM FOR BAGGING LOADED ") 
else:
    print(" CANNOT LOAD 7") 
    exit(0)

classifier_org.append(est_ranFor_bag)

# IMPORT RAND FOREST WITH RANDOM PARAM SEARCH  MODEL # 8
model_name = './model/random_For__fea_no_ext_300_2.pkl' 
if os.path.isfile(model_name): 
    est_ranFor_param = joblib.load(model_name)
    print(" RANDOM FOR PARAM LOADED ") 
else:
    print(" CANNOT LOAD 8 ") 
    exit(0)

classifier_org.append(est_ranFor_param)

# IMPORT FOREST WITH 100 TREES : MODEL # 9

model_name = './model/random_For__fea_no_ext_100.pkl'
if os.path.isfile(model_name): 
    est_ranFor_100 = joblib.load(model_name)
    print(" RANDOM FOREST 100 LOADED ") 
else:
    print(" CANNOT LOAD 9 ") 
    exit(0)

classifier_org.append(est_ranFor_100)

# IMPORT FOREST WITH 200 TREES : MODEL #10
model_name = './model/random_For__fea_no_ext_200.pkl'
if os.path.isfile(model_name): 
    est_ranFor_200 = joblib.load(model_name)
    print(" RANDOM FOREST 200 LOADED ") 
else:
    print(" CANNOT LOAD 10 ") 
    exit(0)

classifier_org.append(est_ranFor_200)

# IMPORT GBR WITH 300 TREES  : MODEL #10 

# IMPORT EXTRA TREE WITH 900 TREES : MODEL #9
print("Completed Fitting Data to Model")
print("Model Info")
print(" Important Features : ") 
print(est.feature_importances_)

print(" Validate Model ")
print("Extra Classifer Model + LINEAR SVC")
pred_valid_y = est.predict(X_valid_red)
testEstimator(est , y_valid_red , pred_valid_y)
preds = est.predict(test_red)

#   print("Classifer Model + LINEAR SVC + Log Reg ")
#   pred_valid_y = est_lg.predict(est_ehc.transform(est.apply(X_valid_red)))
#   testEstimator(est_lg , y_valid_red , pred_valid_y)
#   preds += est_lg.predict(est_ehc.transform(est.apply(test_red))) # Combining the predictions of two classifiers  TAKE MEAN 

#MODEL 3 4 5 6 7 8 
# FORM PREDICTION ON THESE MODELS BY LOOPING THROUGH CLASSIFIER 

for model in range(len(classifier_org)):
  # CHECK AT THE VALIDITY OF THE MODEL 
  print(" VALIDATING MODEL ")
  pred_valid_y = classifier_org[model].predict(X_valid)
  testEstimator(classifier_org[model] , y_valid , pred_valid_y) 
  # FORM PREDICTION FROM THE MODEL 
  print(" FORM PREDICTION FROM THE MODEL  ");
  preds += classifier_org[model].predict(test)

for model in range(len(classifier_lowVar)):
  # CHECK AT THE VALIDITY OF THE MODEL 
  print(" VALIDATING LOW VAR MODEL ")
  pred_valid_y = classifier_lowVar[model].predict(X_valid_var)
  testEstimator(classifier_lowVar[model] , y_valid_var , pred_valid_y) 
  # FORM PREDICTION FROM THE MODEL 
  print(" FORM PREDICTION FROM THE MODEL  ");
  preds += classifier_lowVar[model].predict(test_red_var_1)


# NORMALIZE THE PREDICTED VALUE 
preds /= (1 + len(classifier_lowVar) + len(classifier_org) )

# SAVE THE MODEL 
create_submission(preds, 'mean')



