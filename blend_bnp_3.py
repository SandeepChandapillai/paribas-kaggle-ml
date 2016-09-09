"""
    > Used for extracing the data ... Will change this into mine 
    > Additions using gradient boosting regression technique gbr 
    BLENDING DONE BY USING : ::
https://github.com/emanuele/kaggle_pbr/blob/master/blend.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.cross_validation import StratifiedKFold
from sklearn.linear_model import LogisticRegression

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


X = train ; 
y = labels; 
X_submission = test; 
n_folds = 2

skf = list(StratifiedKFold(y, n_folds))


# BLEND 1
clfs = [
        RandomForestClassifier(n_estimators=100, n_jobs=-1, criterion='gini'),
        RandomForestClassifier(n_estimators=80, max_features = "auto" ,min_samples_split = 30 , n_jobs=-1, criterion='entropy'),
        RandomForestClassifier(n_estimators=150, max_features = 80 ,min_samples_split = 50 , n_jobs=-1, criterion='entropy'),
        RandomForestClassifier(n_estimators=50, max_features = "auto" ,min_samples_split = 70 , n_jobs=-1, criterion='entropy'),
        ExtraTreesClassifier(n_estimators=120, n_jobs=-1, max_depth=50 , max_features = 60 ,  min_samples_leaf=40 , criterion='gini'),
        ExtraTreesClassifier(n_estimators=150, n_jobs=-1, max_depth=100 , max_features = 80 ,  min_samples_leaf=40 , criterion='entropy'),
        ExtraTreesClassifier(n_estimators=100, n_jobs=-1, max_depth=100 , max_features = 30 ,  min_samples_leaf=30 , criterion='entropy'),
        ExtraTreesClassifier(n_estimators=150, n_jobs=-1, max_depth=120 , max_features = "auto" ,  min_samples_leaf=20 , criterion='entropy')
        ]

print "Creating train and test sets for blending."

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))


from sklearn.externals import joblib
for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        print "Fold", i
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:,1]
        dataset_blend_train[test, j] = y_submission
        dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        joblib.dump(clf, './model/' + 'blend_' +str(j) + '.pkl')
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)
    joblib.dump(clf, './model/' + 'blend_final_2' +str(j) + '.pkl')

print
print "Blending."
#clf = LogisticRegression(verbose = 1 , n_jobs = -1 )

clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, max_depth=20 , max_features = "auto" ,  min_samples_leaf=40 , criterion='gini')

clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:,1]

joblib.dump(clf, './model/' + 'blend_final_3.pkl')

print "Linear stretch of predictions to [0,1]"
y_submission = (y_submission - y_submission.min()) / (y_submission.max() - y_submission.min())

preds = y_submission;

print(preds)
print("Obtained Prediction")
print("Creating csv file")
submission = pd.read_csv('sample_submission.csv')
submission["PredictedProb"] = preds
submission.to_csv('submission.csv', index=False)

print("Saving the Model")

# To load the model 
#est = joblib.load('extraTree.pkl')

