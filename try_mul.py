"""

BAYSIAN RIDGE 


"""



import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import VarianceThreshold

from sklearn import linear_model 
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from sklearn.lda import LDA 
from sklearn.qda import QDA
from sklearn.kernel_ridge import KernelRidge 
from sklearn import svm 

from sklearn.neighbors import NearestNeighbors

from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn import gaussian_process
from sklearn.ensemble import AdaBoostClassifier

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


def testEstimator(est , y_valid , X_valid):
    pred_valid_y = est.predict(X_valid)
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

print("Cleaning and formatting data")

train = np.asarray(train, dtype=np.float32)        
labels = labels.ravel()

# Convert the data frame column into a single dim array 
print("Creating training set and validation set : 90 / 10 ")
X_train , X_valid , y_train , y_valid  = train_test_split(train , labels , test_size = 0.10 , random_state = 0 )


####################################################################################################

## PIPELINES : 
pipe_poly_lst_sqr =  Pipeline([('poly' , PolynomialFeatures(degree = 10) ) , 
                               ('linear', linear_model.LinearRegression())])

classifiers = [
        ("bays_ridge" , linear_model.BayesianRidge() ) , 
        ("linear_reg" , linear_model.LinearRegression() ) , 
        ("ransac" , linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold = 0.5 ) ),
        ("sgd" , linear_model.SGDClassifier()) , 
        ("asgd" , linear_model.SGDClassifier(average = True)) , 
        ("preception" , linear_model.Perceptron() ) , 
        ("passive_aggres_1", linear_model.PassiveAggressiveClassifier(loss='hinge', C=1.0)),
        ("passive_aggres_2", linear_model.PassiveAggressiveClassifier(loss='squared_hinge',C=1.0)),
     #   ("poly_lst_squ" ,pipe_poly_lst_sqr )  , 
        ("lda" , LDA()) , 
        ("qda" , QDA()) , 
        ("kernel_ridge" , KernelRidge(alpha = 1.0) ) , 
        ("svm" , svm.SVC()) , 
        ("svr" , svm.SVR()) , 
        ("svm_rbf" , svm.SVC(kernel='rbf' , gamma = 0.2)) , 
        ("nearest_neigh" , NearestNeighbors(n_neighbors = 10 , algorithm='auto')),
        ("gaus_nai_bayes" , GaussianNB()) ,
        ("log_reg" , linear_model.LogisticRegression()) ,
        ("descision_tree" , tree.DecisionTreeClassifier()), 
        ("gaussian_proc" , gaussian_process.GaussianProcess()),
        ("descision_tree" ,tree.DecisionTreeClassifier(max_depth=5)),
        ("ada_boost", AdaBoostClassifier() ) , 
        ]



print("Creating model")

for i in (classifiers):
    print(i[0])
    print("Fitting data")
    i[1].fit(X_train,y_train)
    print("Completed Fitting Data to Model")
    testEstimator(i[1] , y_valid , X_valid)
    print("Saving the Model")
    joblib.dump( i[1] , './model/' + str(i[0]))




#
#   model = " BAYSIAN RIDGE REGRESSION " ; 
#   print(model)
#   est = linear_model.BayesianRidge() # Fits the data better ?? 
#
#
#   model = "LINEAR REGRESSOR"
#   print(model)
#   est = linear_model.LinearRegression() # Fits the data better ?? 
#   print("Fitting data")
#   est.fit(X_train,y_train)
#   print("Completed Fitting Data to Model")
#   testEstimator(est , y_valid , X_valid)
#   print("Saving the Model")
#   joblib.dump(est, './model/' + str(model))
#
#
#   model = "RANSAC"
#   print(model)
#
#   est = linear_model.RANSACRegressor(linear_model.LinearRegression())
#
#   print("Fitting data")
#   est.fit(X_train,y_train)
#   print("Completed Fitting Data to Model")
#   testEstimator(est , y_valid , X_valid)
#   print("Saving the Model")
#   joblib.dump(est, './model/' + str(model))
#
#


