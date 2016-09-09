"""
Created on Tue Feb 23 12:01:21 2016

@author: Ouranos
    > Used for extracing the data ... Will change this into mine 

    > Additions using gradient boosting regression technique gbr 

"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor as algo
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler


# new way of extracting data 

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



class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = np.float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)


def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()       
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler


        
def getDummiesInplace(columnList, train, test = None):
    #Takes in a list of column names and one or two pandas dataframes
    #One-hot encodes all indicated columns inplace
    columns = []
    
    if test is not None:
        df = pd.concat([train,test], axis= 0)
    else:
        df = train
        
    for columnName in df.columns:
        index = df.columns.get_loc(columnName)
        if columnName in columnList:
            dummies = pd.get_dummies(df.ix[:,index], prefix = columnName, prefix_sep = ".")
            columns.append(dummies)
        else:
            columns.append(df.ix[:,index])
    df = pd.concat(columns, axis = 1)
    
    if test is not None:
        train = df[:train.shape[0]]
        test = df[train.shape[0]:]
        return train, test
    else:
        train = df
        return train
        
def pdFillNAN(df, strategy = "mean"):
    #Fills empty values with either the mean value of each feature, or an indicated number
    if strategy == "mean":
        return df.fillna(df.mean())
    elif type(strategy) == int:
        return df.fillna(strategy)



train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

np.random.seed(3210)
train = train.iloc[np.random.permutation(len(train))]

#Drop target, ID, and v22(due to too many levels), and high correlated columns
labels = train["target"]
trainId = train["ID"]
testId = test["ID"]

#train.drop(["ID","target","v22","v107","v71","v31","v100","v63","v64"])
train.drop(labels = ['ID','target',"v22",'v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1, inplace = True)

#train.drop(['ID','target',"v22"],axis=1, inplace = True)

#test.drop(labels = ["ID","v22","v107","v71","v31","v100","v63","v64"], axis = 1, inplace = True)
test.drop(labels = ["ID","v22",'v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1, inplace = True)

#test.drop(labels = ["ID","v22"], axis =1 , inplace = True)

#find categorical variables
categoricalVariables = []
for var in train.columns:
    vector=pd.concat([train[var],test[var]], axis=0)
    typ=str(train[var].dtype)
    if (typ=='object'):
        categoricalVariables.append(var)


print ("Generating dummies...")
train, test = getDummiesInplace(categoricalVariables, train, test)

#Remove sparse columns
cls = train.sum(axis=0)
train = train.drop(train.columns[cls<10], axis=1)
test = test.drop(test.columns[cls<10], axis=1)

print ("Filling in missing values...")
fillNANStrategy = -1
#fillNANStrategy = "mean"
train = pdFillNAN(train, fillNANStrategy)
test = pdFillNAN(test, fillNANStrategy)


print ("Scaling...")
train, scaler = preprocess_data(train)
test, scaler = preprocess_data(test, scaler)

print("Cleaning and formatting data")

train = np.asarray(train, dtype=np.float32)        
labels = np.asarray(labels, dtype=np.int32).reshape(-1,1)


# Convert the data frame column into a single dim array 
print("Creating training set and validation set : 90 / 10 ")
X_train , X_valid , y_train , y_valid  = train_test_split(train , labels , test_size = 0.10 , random_state = 0 )


print("Creating model")
est = algo(n_estimators=1200,max_features= 30,criterion= 'entropy',min_samples_split= 2, max_depth= 30, min_samples_leaf= 2, n_jobs = -1)


print("Fitting data")
est.fit(X_train,y_train)
print("Completed Fitting Data to Model")

print("Model Info")
print(" Important Features : ") 
print(est.feature_importances_)

print(" Validate Model ")
pred_valid_y = est.predict(X_valid)

testEstimator(est , y_valid , pred_valid_y)

preds = est.predict(test)

create_submission(preds, "")

print("Saving the Model")
from sklearn.externals import joblib
joblib.dump(est, './model/extraTree__fea_ext_qu_1200.pkl')


