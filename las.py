"""
 Using the mlp example of lasagne to train the network 


https://raw.githubusercontent.com/Lasagne/Lasagne/master/examples/mnist.py

"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score 

import theano
import theano.tensor as T

import lasagne


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

#Remove sparse columns
print("Removing Spare Columns")
cls = train.sum(axis=0)
train = train.drop(train.columns[cls<10], axis=1)
test = test.drop(test.columns[cls<10], axis=1)


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

print(" Creating Neural Network Model ")

train_var = T.tensor('inputs')
target_var = T.ivector('targets')

l_in = lasagne.layers.InputLayer(shape=(None , 1 ,131), input_var = train_var) 


l_in_drop = lasagne.layers.DropoutLayer(l_in , p = 0.2)

l_hid1 = lasagne.layers.DenseLayer(l_in_drop , num_units = 300 , 
                        nonlinearity=lasagne.nonlinearities=rectify,
                        W=lasagne.init.GlorotUniform())

l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1 , p = 0.5)


l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop , num_units = 300 , nonlinearity = lasagne.nonlinearities.rectify)



l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2 , p = 0.5)


l_out = lasagne.layers.DenseLayer(l_hid2_drop , num_units = 1 , nonlinearity = lasagne.nonlinearities.softmax)
















