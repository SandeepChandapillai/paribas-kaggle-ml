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


from sklearn import gaussian_process
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KDTree
from sklearn.linear_model import SGDClassifier


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


train.drop(["v22",'v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1, inplace = True)


test.drop(labels = ["v22",'v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1, inplace = True)


train = np.asarray(train, dtype=np.float32)        
labels = labels.ravel()



X = train ; 
y = labels; 
X_submission = test; 
n_folds = 2

skf = list(StratifiedKFold(y, n_folds))

from sklearn.externals import joblib
from sklearn import svm

from sklearn.linear_model import LinearRegression

from sklearn import linear_model
# BLEND 1
clfs = [
        #gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1),
        NearestCentroid(),
        SGDClassifier( penalty="l2"),
        svm.SVC(kernel='rbf'),
        LinearRegression(fit_intercept=False),
        linear_model.RANSACRegressor(linear_model.LinearRegression())
        ]

n_new_models = len(clfs)
#Add the saved models 

#pre_clfs = []
#blend_final_red
# blend_red4.pkl 
model_name = "./model/blend_final_red" 
for i in range(8):
   name = model_name + (str(i) + ".pkl") 
   model = joblib.load(name)
   clfs.append(model)



print "Creating train and test sets for blending."

dataset_blend_train = np.zeros((X.shape[0], len(clfs)))
dataset_blend_test = np.zeros((X_submission.shape[0], len(clfs)))

for j, clf in enumerate(clfs):
    print j, clf
    dataset_blend_test_j = np.zeros((X_submission.shape[0], len(skf)))
    for i, (train, test) in enumerate(skf):
        print "Fold", i
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]
        if j < n_new_models: # fit data if it is a new model 
            clf.fit(X_train, y_train)
        y_submission = clf.predict_proba(X_test)[:,1]
        dataset_blend_train[test, j] = y_submission
        if j < n_new_models:
            dataset_blend_test_j[:, i] = clf.predict(X_submission)
        else:
            dataset_blend_test_j[:, i] = clf.predict_proba(X_submission)[:,1]
        if j < n_new_models:
            joblib.dump(clf, './model/' + 'blend_red_2_' +str(j) + '.pkl')
    dataset_blend_test[:,j] = dataset_blend_test_j.mean(1)


print
print "Blending."
#clf = LogisticRegression(verbose = 1 , n_jobs = -1 )

clf = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, max_depth=20 , max_features = "auto" ,  min_samples_leaf=40 , criterion='gini')

clf.fit(dataset_blend_train, y)
y_submission = clf.predict_proba(dataset_blend_test)[:,1]

joblib.dump(clf, './model/' + 'blend_final_red_2.pkl')

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

