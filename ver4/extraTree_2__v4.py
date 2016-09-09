import pandas as pd
import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble

print('Load data...')
train = pd.read_csv("train.csv")
target = train['target'].values
trainId = train.pop("ID")
train = train.drop(['target'],axis=1)

train = train.drop(['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

test = pd.read_csv("test.csv")
testId = test.pop("ID")
test = test.drop(['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999

from sklearn.externals import joblib
print("Opening model")
est = joblib.load('./model/extraTree__fea_ext_ka2_750.pkl')
#        'extraTree__fea_ext_ka2_1200.pkl',

print("Saving the response from train set")
y_pred = est.predict(train)
pd.DataFrame({"ID": trainId, "PredictedProb": y_pred }).to_csv('./ver4/train_exT_ds_1_750.csv' , index = False)

print("Saving the response from test set")
y_pred = est.predict(test)
pd.DataFrame({"ID": testId, "PredictedProb": y_pred }).to_csv('./ver4/test_exT_ds_1_750.csv', index=False)
