import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier


def readTest(model_name):
    submission = pd.read_csv('./ver4/res/test_' + model_name + '.csv')
    return submission["PredictedProb"]

def readTrain(model_name):
    submission = pd.read_csv('./ver4/res/train_' + model_name + '.csv')
    return submission["PredictedProb"]

from ensem_stage_0 import getDataSet_1

print("LOADING DATA")
train , label , test = getDataSet_1()

models_name = [
    'ada_ds_2_200',
    'ada_ds_2_300',
    'ada_ds_2_400',
    'exT_ds_1_1200',
    'exT_ds_12_710',
    'exT_ds_1_750',
    'net_ds_2_250_120_50_25',
    'net_ds_2_250_150_80_25__100',
    'net_ds_2_250_150_80_25'
    ]



blend_train = np.zeros((train.shape[0], len(models_name)))

blend_test = np.zeros((test.shape[0], len(models_name)))

print("Predicting ....")
for i in range(len(models_name)):
    print("Model : " + models_name[i] + " " + str(i))
    blend_train[:, i] = readTrain(models_name[i])
    print(blend_train[:, i])
    blend_test[:, i] = readTest(models_name[i])
    print(blend_test[:, i])

blend_est = ExtraTreesClassifier(n_estimators=200, n_jobs=-1, max_depth=20 , max_features = "auto" ,  min_samples_leaf=40 , criterion='gini')
blend_est.fit(blend_train , label)
preds = blend_est.predict_proba(blend_test)[:,0]

print(preds)
print("Obtained Prediction")
print("Creating csv file")
submission = pd.read_csv('sample_submission.csv')
submission["PredictedProb"] = preds
submission.to_csv('blend_ext_200.csv', index=False)









