"""
Created on Tue Feb 23 12:01:21 2016

@author: Ouranos
    > Used for extracing the data ... Will change this into mine 


    > Additions using gradient boosting regression technique gbr 

"""


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

   

# replace categorial values by splitting it into multiple integer columns         
# say strings a b c in col v1 , then create 3 new columns in place of v1 , with
#  v1   ->    va   vb  vc 
#   a          1    0   0
#   b          0    1   0
#   c          0    0   1

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


#Drop target, ID, and v22(due to too many levels), and high correlated columns
labels = train["target"]
trainId = train["ID"]
testId = test["ID"]


train.drop(['ID','target',"v22"],axis=1, inplace = True)

test.drop(labels = ["ID","v22"], axis =1 , inplace = True)

#find categorical variables
categoricalVariables = []
for var in train.columns:
    vector=pd.concat([train[var],test[var]], axis=0)
    typ=str(train[var].dtype)
    if (typ=='object'):
        categoricalVariables.append(var)


print ("Converting Strings into integer columns ...")
train, test = getDummiesInplace(categoricalVariables, train, test)


print ("Filling in missing values...")
fillNANStrategy = -1
#fillNANStrategy = "mean"
train = pdFillNAN(train, fillNANStrategy)
test = pdFillNAN(test, fillNANStrategy)

print("Cleaning and formatting data")

train = np.asarray(train, dtype=np.float32)        
labels = labels.ravel()

# Convert the data frame column into a single dim array 

print("Creating model")
#est = GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.1,
#                                          loss='huber', min_samples_leaf=1, 
#                                          random_state=0)

est = GradientBoostingClassifier(n_estimators=1000, max_depth=10 , min_samples_leaf=100)

print("Fitting data")
est.fit(train,labels)
print("Completed Fitting Data to Model")

preds = est.predict_proba(test)
print(preds[:,1])

submission = pd.read_csv('sample_submission.csv')
submission["PredictedProb"] = preds[:,1]
submission.to_csv('submission.csv', index=False)


