
"""

Use ensemble of 3 models.

start small for now...


APRIL 6TH..
Have hope...

BEST MODELS :

    extra trees
    xgboost 
#    randfor 

"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import mean_squared_error


from ensem_stage_0 import getDataSet_1 

train1 , label1 , test1 = getDataSet_1()

from sklearn.externals import joblib

print("##### LOADING MODEL #######")
model1_1 = joblib.load('./model/extraTree__fea_ext_ka2_750.pkl')
model1_2 = joblib.load('./model/extraTree__fea_ext_ka2_1200.pkl')

print("##### COMPLETED LOADING MODEL ######")
#### CREATE ENSEMBLE #### 

print(train1.shape[0])
num_models = 2
blend_train = np.zeros((train1.shape[0] , num_models))
blend_test = np.zeros((test1.shape[0] , num_models)) 

print("Predicting....")
blend_train[: , 0] = model1_1.predict_proba(train1)[:,1]
blend_train[: , 1] = model1_2.predict_proba(train1)[:,1]
blend_test[: , 0] = model1_1.predict_proba(test1)[:,1]
blend_test[: , 1] = model1_2.predict_proba(test1)[:,1]


print("Predicting....Test set")
blend_est  = ExtraTreesClassifier( n_jobs=-1)

blend_est.fit(blend_train , label1)

preds = blend_est.predict_proba(blend_test)[:,1]

print(preds)
print("Obtained Prediction")
print("Creating csv file")
submission = pd.read_csv('sample_submission.csv')
submission["PredictedProb"] = preds
submission.to_csv('submission.csv', index=False)























