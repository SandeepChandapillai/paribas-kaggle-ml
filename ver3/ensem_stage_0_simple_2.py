
"""

Use ensemble of 3 models.

start small for now...


APRIL 6TH..
Have hope...

BEST MODELS :

    extra trees
    xgboost 
#    randfor 

Let all the models be classifiers : I know how to use them 
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import mean_squared_error


from sklearn.linear_model import LinearRegression

from ensem_stage_0 import getDataSet_1 

train1 , label1 , test1 = getDataSet_1()

from sklearn.externals import joblib

print("##### LOADING MODEL #######")
model1_1 = joblib.load('./model/extraTree__fea_ext_ka2_750.pkl')
#model1_2 = joblib.load('./model/extraTree__fea_ext_ka2_1200.pkl')
model1_2 = joblib.load('./model/linear_reg__dataSet1')
model1_3 = joblib.load('./model/svr__dataSet1')

print("##### COMPLETED LOADING MODEL ######")
#### CREATE ENSEMBLE #### 

clfs = [ model1_1 , model1_2 , model1_3]

print(train1.shape[0])
num_models = len(clfs)
blend_train = np.zeros((train1.shape[0] , num_models))
blend_test = np.zeros((test1.shape[0] , num_models)) 

print("Predicting....")
for i in range(num_models):
        print("Model " + str(i))
	blend_train[: , i] = clfs[i].predict(train1)
        print(blend_train[:,i])
	blend_test[: , i] = clfs[i].predict(test1)
        print(blend_test[:,i])

print("Predicting....Test set")
blend_est   = ExtraTreesClassifier(n_estimators=100, n_jobs=-1, max_depth=20 , max_features = "auto" ,  min_samples_leaf=40 , criterion='gini')
blend_est.fit(blend_train , label1)
preds = blend_est.predict(blend_test)

print(preds)
print("Obtained Prediction")
print("Creating csv file")
submission = pd.read_csv('sample_submission.csv')
submission["PredictedProb"] = preds
submission.to_csv('submission.csv', index=False)





















