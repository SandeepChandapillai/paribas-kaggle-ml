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
    print(pred_valid_y)
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


print('Load data...')
train = pd.read_csv("train.csv")
target = train['target'].values
train = train.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
test = pd.read_csv("test.csv")
id_test = test['ID'].values
test = test.drop(['ID','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

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

labels = target
train = np.asarray(train, dtype=np.float32)        

# Convert the data frame column into a single dim array 
print("Creating training set and validation set : 90 / 10 ")
X_train , X_valid , y_train , y_valid  = train_test_split(train , labels , test_size = 0.10 , random_state = 0 )


####################################################################################################

## PIPELINES : 
pipe_poly_lst_sqr =  Pipeline([('poly' , PolynomialFeatures(degree = 10) ) , 
                               ('linear', linear_model.LinearRegression())])

classifiers = [
       # ("bays_ridge" , linear_model.BayesianRidge() ) , 
       # ("linear_reg" , linear_model.LinearRegression() ) , 
   #     ("ransac" , linear_model.RANSACRegressor(linear_model.LinearRegression(), residual_threshold = 0.5 ) ),
       # ("sgd" , linear_model.SGDClassifier()) , 
       # ("asgd" , linear_model.SGDClassifier(average = True)) , 
       # ("preception" , linear_model.Perceptron() ) , 
       # ("passive_aggres_1", linear_model.PassiveAggressiveClassifier(loss='hinge', C=1.0)),
       # ("passive_aggres_2", linear_model.PassiveAggressiveClassifier(loss='squared_hinge',C=1.0)),
     #   ("poly_lst_squ" ,pipe_poly_lst_sqr )  , 
       # ("lda" , LDA()) , 
       # ("qda" , QDA()) , 
        #("kernel_ridge" , KernelRidge(alpha = 1.0) ) , 
       # ("svm" , svm.SVC()) , 
       # ("svr" , svm.SVR()) , 
       # ("svm_rbf" , svm.SVC(kernel='rbf' , gamma = 0.2)) , 
       # ("nearest_neigh" , NearestNeighbors(n_neighbors = 10 , algorithm='auto')),
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
    joblib.dump( i[1] , './model/' + str(i[0]) + '__dataSet1')




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


