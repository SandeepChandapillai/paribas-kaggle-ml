import numpy as np
import theano
import theano.tensor as T # for shared dataset

import math # isnan 


def con_str_int(s):
    num = 0 ;  
    for i in s:
        num += ord(i)
    return num

# num_data determines how many triaing examples to read from 
"""
 Reads the data from the test set 

"""
def readData_bnp():
     data = np.recfromcsv('train.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
    

    # extract the result => y
     target = np.array([])
     for i in data:
        # go over each data element and extract the 1 th element
         target = np.append(target,i[1])

    # extract the input = x
     input_x = [] # create a list and then add elements to this list and covert to np.array
     val = 0
     for i in data:
         temp = []
         for j in range(2,len(i)):
             # we convert the strings to int.. 
             val = i[j]
             #val = np.nan_to_num(val) # convert nan to int ?  output is nd_array causes problems
             if isinstance(val , (str , unicode)):
                 val = float(con_str_int(val)) # converts the string to int ?      
                 #print val 
             if np.isnan(val):
                 val = 0.0 
             temp.append(val) # add val to list 


         input_x.append(temp) # add the temp to input
    
     # create training set and validation set
     # training st 90% validation set 10%
     n_train = len(input_x) * 0.9
     n_train = int(n_train) # covert to int

     input_x = np.matrix(input_x)
 
     #np.nan_to_num(input_x) # handle nan values ? 
     train_x = input_x[:n_train]
     train_y = target[:n_train]

     # validation set
     valid_x = input_x[n_train:]
     valid_y = target[n_train:]

     # this function outputs the train set and validation set and testing set 
     train_xy = [train_x , train_y]
     valid_xy = [valid_x , valid_y]

     #return train_xy
     # covert into theano objects ?? 
     def shared_dataset(data_xy, borrow=True):
         data_x, data_y = data_xy
         shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
         shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
         return shared_x, T.cast(shared_y, 'int32')

     print("Create theano objects ... ")
     #test_set_x, test_set_y = shared_dataset(test_xy)
     valid_set_x, valid_set_y = shared_dataset(valid_xy)
     train_set_x, train_set_y = shared_dataset(train_xy)
  #   test_x = theano.shared(np.asarray(input_test_x,
  #                                             dtype=theano.config.floatX),
  #                               borrow=True)
     rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
     return rval


'''
    0 index gives the ids 
    1 index gives the data 
'''

def readTest_bnp():
     data = np.recfromcsv('test.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
     input_test_x = [] # create a list and then add elements to this list and covert to np.array
     id_test = [] # handles the id , To write to kaggle output file 
     for i in data:
         id_test.append(i[0]);
         temp = []
         for j in range(1,len(i)): # test set does not have the target column , so start reading from the index 1 column , 
             # we convert the strings to int.. 
             val = i[j]
             #val = np.nan_to_num(val) # convert nan to int ? 
             if isinstance(val , (str , unicode)):
                 val = float(con_str_int(val)) # converts the string to int ?      
             if np.isnan(val):
                 val = 0.0 ;

             temp.append(val) # add val to list 
         input_test_x.append(temp) # add the temp to input

     input_test_x = np.matrix(input_test_x)
     test_x = theano.shared(np.asarray(input_test_x,dtype=theano.config.floatX),borrow=True)

     return [id_test ,test_x]; 


'''
     print(" Reading in test data ")
     # readin the test set
     data_test = np.recfromcsv('test.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
     input_test_x = [] # create a list and then add elements to this list and covert to np.array
     for i in data:
         temp = []
         for j in range(1,len(i)): # test set does not have the target column , so start reading from the index 1 column , 
             # we convert the strings to int.. 
             val = i[j]
             #val = np.nan_to_num(val) # convert nan to int ? 
             if isinstance(val , (str , unicode)):
                 val = float(con_str_int(val)) # converts the string to int ?      
             if np.isnan(val):
                 val = 0.0 ;

             temp.append(val) # add val to list 
         input_test_x.append(temp) # add the temp to input
     input_test_x = np.matrix(input_test_x)
     #np.nan_to_num(input_test_x) # handle nan values ? 
'''

"""

Read reduced number of training sets

"""

def readDataRed_bnp(num = 0):
     data = np.recfromcsv('train.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')

    # extract the result => y
     target = np.array([])
     for i in data:
        # go over each data element and extract the 1 th element
         target = np.append(target,i[1])

    # extract the input = x
     input_x = [] # create a list and then add elements to this list and covert to np.array
     val = 0
     if num == 0 :
         num = len(data)
     
     for i in range(num):
         temp = []
         for j in range(2,len(data[i])):
             # we convert the strings to int.. 
             val = data[i][j]
             #val = np.nan_to_num(val) # convert nan to int ?  output is nd_array causes problems
             if isinstance(val , (str , unicode)):
                 val = float(con_str_int(val)) # converts the string to int ?      
                 #print val 
             if np.isnan(val):
                 val = 0.0 
             temp.append(val) # add val to list 
         input_x.append(temp) # add the temp to input
    
     # create training set and validation set
     # training st 90% validation set 10%
     n_train = len(input_x) * 0.9
     n_train = int(n_train) # covert to int

     input_x = np.matrix(input_x)
 
     #np.nan_to_num(input_x) # handle nan values ? 
     train_x = input_x[:n_train]
     train_y = target[:n_train]

     # validation set
     valid_x = input_x[n_train:]
     valid_y = target[n_train:]

     # this function outputs the train set and validation set and testing set 
     train_xy = [train_x , train_y]
     valid_xy = [valid_x , valid_y]

     #return train_xy
     # covert into theano objects ?? 
     def shared_dataset(data_xy, borrow=True):
         data_x, data_y = data_xy
         shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
         shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
         return shared_x, T.cast(shared_y, 'int32')

     print("Create theano objects ... ")
     #test_set_x, test_set_y = shared_dataset(test_xy)
     valid_set_x, valid_set_y = shared_dataset(valid_xy)
     train_set_x, train_set_y = shared_dataset(train_xy)
  #   test_x = theano.shared(np.asarray(input_test_x,
  #                                             dtype=theano.config.floatX),
  #                               borrow=True)
     rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
     return rval



"""


Reduced Number + No Share 

"""
def readDataRS_bnp(num = 0):
     data = np.recfromcsv('train.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')

    # extract the result => y
     target = np.array([])
     for i in data:
        # go over each data element and extract the 1 th element
         target = np.append(target,i[1])

    # extract the input = x
     input_x = [] # create a list and then add elements to this list and covert to np.array
     val = 0
     if num == 0 :
         num = len(data)
     
     for i in range(num):
         temp = []
         for j in range(2,len(data[i])):
             # we convert the strings to int.. 
             val = data[i][j]
             #val = np.nan_to_num(val) # convert nan to int ?  output is nd_array causes problems
             if isinstance(val , (str , unicode)):
                 val = float(con_str_int(val)) # converts the string to int ?      
                 #print val 
             if np.isnan(val):
                 val = 0.0 
             temp.append(val) # add val to list 
         input_x.append(temp) # add the temp to input
    
     # create training set and validation set
     # training st 90% validation set 10%
     n_train = len(input_x) * 0.9
     n_train = int(n_train) # covert to int
     
     print("Array Size + Types")
     print(type(input_x))
     #print(input_x)
     #print(input_x[0])

     input_x = np.array(input_x)
     target = np.array(target) 

     #np.nan_to_num(input_x) # handle nan values ? 
     train_x = input_x[:n_train]
     train_y = target[:n_train]

     # validation set
     valid_x = input_x[n_train:]
     valid_y = target[n_train:]

     # this function outputs the train set and validation set and testing set 
     train_xy = [train_x , train_y]
     valid_xy = [valid_x , valid_y]

     rval = [train_xy , valid_xy]
    
     print("RESULT")
     print(type(rval))
     print(type(rval[0]))
     print(type(rval[1]))
     print(type(rval[0][0]))
     print(type(rval[0][1]))
     print(type(rval[1][0]))
     print(type(rval[1][1]))
     print(rval[1][1])
     print(rval[1][0].shape[0])
     print(rval[1][0].shape[1])



     return rval 



def readTest_N_bnp(num = 0):
     data = np.recfromcsv('test.csv', delimiter=',', filling_values=np.nan, case_sensitive=True, deletechars='', replace_space=' ')
     input_test_x = [] # create a list and then add elements to this list and covert to np.array
     id_test = [] # handles the id , To write to kaggle output file 

     if num == 0 :
         num = len(data)

     for i in range(num):
         id_test.append(data[i][0]);
         temp = []
         for j in range(1,len(data[i])): # test set does not have the target column , so start reading from the index 1 column , 
             # we convert the strings to int.. 
             val = data[i][j]
             #val = np.nan_to_num(val) # convert nan to int ? 
             if isinstance(val , (str , unicode)):
                 val = float(con_str_int(val)) # converts the string to int ?      
             if np.isnan(val):
                 val = 0.0 ;

             temp.append(val) # add val to list 
         input_test_x.append(temp) # add the temp to input

     input_test_x = np.array(input_test_x)
     #test_x = theano.shared(np.asarray(input_test_x,dtype=theano.config.floatX),borrow=True)
     test_x = input_test_x

     rval = [id_test ,test_x]; 

     print("RESULT TEST SET READING")
     print(type(rval))
     print(type(rval[0]))
     print(type(rval[1]))
     print(type(rval[0][0]))
     print(type(rval[0][1]))
     print(type(rval[1][0]))
     print(type(rval[1][1]))
     print(rval[1][1])
     #print(rval[1][0].shape[0])
     #print(rval[1][0].shape[1])



     return rval

     
