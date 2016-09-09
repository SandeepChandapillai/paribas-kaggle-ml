"""
Use multiple data sets ensemble them together.. 


ver 0 :
    org data set 


"""




import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor as algo
from sklearn.cross_validation import train_test_split 
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import mean_squared_error

from sklearn.feature_selection import VarianceThreshold


"""
Data set using my conversion 

"""
def getDataSet_0():
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

    train = np.asarray(train, dtype=np.float32)        
    labels = labels.ravel()

    return (train , labels , test)

"""
KAGGLE DATA SET : USED FOR THE EXTRA TREES WITH HIGH SCORES .... 
"""
def getDataSet_1():
    print('Load data...')
    train = pd.read_csv("train.csv")
    target = train['target'].values
    train = train.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
    test = pd.read_csv("test.csv")
    id_test = test['ID'].values
    test = test.drop(['ID','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

    print('Clearing...')
    ## EXTREMELY SLOW 
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

    return (train , target , test)


"""
KAGGLE DATA SET : USED FOR THE EXTRA TREES WITH HIGH SCORES .... 

Without feature removal
"""
def getDataSet_1_2():

    print('Load data...')
    train = pd.read_csv("train.csv")
    target = train['target'].values
    train = train.drop(['ID','target'],axis=1)
    test = pd.read_csv("test.csv")
    id_test = test['ID'].values
    test = test.drop(['ID'],axis=1)

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

    return (train , target , test)


"""

Data set with 0.3 variance reduced 

"""
def getDataSet_2():
    #data = getDataSet_0()
    #train = data[0]
    #test = data[2]
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # label is y 
    labels = train.pop("target") 
    trainId = train.pop("ID")
    testId = test.pop("ID")

    print ("Filling in missing values...")
    train.fillna(0, inplace = True)
    test.fillna(0 , inplace = True)

    print ("Converting Strings into integer columns ...")
    train = train.applymap(lambda s: con_str_int(s) if isinstance(s , (str , unicode)) else s)
    test = test.applymap(lambda s: con_str_int(s) if isinstance(s , (str , unicode)) else s)

    print("PreProcessing Data")
    train = preprocessing.normalize(train)
    test = preprocessing.normalize(test)

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

    return (train , labels , test)



"""

Data set with 0.6 variance reduced 

"""
def getDataSet_2_2():
    #data = getDataSet_0()
    #train = data[0]
    #test = data[2]
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    # label is y 
    labels = train.pop("target") 
    trainId = train.pop("ID")
    testId = test.pop("ID")

    print ("Filling in missing values...")
    train.fillna(0, inplace = True)
    test.fillna(0 , inplace = True)

    print ("Converting Strings into integer columns ...")
    train = train.applymap(lambda s: con_str_int(s) if isinstance(s , (str , unicode)) else s)
    test = test.applymap(lambda s: con_str_int(s) if isinstance(s , (str , unicode)) else s)

    print("PreProcessing Data")
    train = preprocessing.normalize(train)
    test = preprocessing.normalize(test)

    print("Input Data set :")
    print(train.describe())

    #Remove Columns with low variance :
    print("Removing Columns with Low Variance")
    fea_sel = VarianceThreshold(threshold = 0.6)
    train = fea_sel.fit_transform(train)
    test = fea_sel.fit_transform(test)

    print("Cleaning and formatting data")

    train = np.asarray(train, dtype=np.float32)        
    labels = labels.ravel()

    return (train , labels , test)


"""

Data set with compression 

"""
def dataDataSet_2_3():
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
    fea_sel = VarianceThreshold(threshold = 0.6)
    train_red_var = fea_sel.fit_transform(train)
    test_red_var = fea_sel.fit_transform(test)


    print("Selecting the best Features using Linear SVM ")
    model_name_svc = 'model/linear_svc_pre_l1.pkl'
    model_exist_svc = False 
    if os.path.isfile(model_name_svc): 
        print('Model Exist')
        lsvc = joblib.load(model_name_svc)
        model_exist_svc = True 
    else:
        lsvc = LinearSVC(C=0.1 , penalty = "l1" , dual = False).fit(train , labels)

    if not (model_exist_svc):
        print("Saving the Model")
        joblib.dump(lsvc , model_name_svc)

    model = SelectFromModel(lsvc , prefit = True)
    train_red = model.transform(train_red_var)
    test_red = model.transform(test_red_var)
    # Some models use the entire model Others use a reduced portion of the model 



"""

Data set used for the neural network 

"""
def getDataSet_3():
    class AdjustVariable(object):
        def __init__(self, name, start=0.03, stop=0.001):
            self.name = name
            self.start, self.stop = start, stop
            self.ls = None

        def __call__(self, nn, train_history):
            if self.ls is None:
                self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

            epoch = train_history[-1]['epoch']
            new_value = np.float32(self.ls[epoch - 1])
            getattr(nn, self.name).set_value(new_value)


    def preprocess_data(X, scaler=None):
        if not scaler:
            scaler = StandardScaler()       
            scaler.fit(X)
        X = scaler.transform(X)
        return X, scaler


            
    def getDummiesInplace(columnList, train, test = None):
        #Takes in a list of column names and one or two pandas dataframes
        #One-hot encodes all indicated columns inplace
        columns = []
        
        if test is not None:
            df = pd.concat([train,test], axis= 0)
        else:
            df = train
            
        for columnName in df.columns:
            index = df.columns.get_loc(columnName)
            if columnName in columnList:
                dummies = pd.get_dummies(df.ix[:,index], prefix = columnName, prefix_sep = ".")
                columns.append(dummies)
            else:
                columns.append(df.ix[:,index])
        df = pd.concat(columns, axis = 1)
        
        if test is not None:
            train = df[:train.shape[0]]
            test = df[train.shape[0]:]
            return train, test
        else:
            train = df
            return train
            
    def pdFillNAN(df, strategy = "mean"):
        #Fills empty values with either the mean value of each feature, or an indicated number
        if strategy == "mean":
            return df.fillna(df.mean())
        elif type(strategy) == int:
            return df.fillna(strategy)



    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    np.random.seed(3210)
    train = train.iloc[np.random.permutation(len(train))]

    #Drop target, ID, and v22(due to too many levels), and high correlated columns
    labels = train["target"]
    trainId = train["ID"]
    testId = test["ID"]

    #train.drop(labels = ["ID","target","v22","v107","v71","v31","v100","v63","v64"], axis = 1, inplace = True)
    train.drop(['ID','target',"v22",'v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1, inplace = True)

    #train.drop(['ID','target',"v22"],axis=1, inplace = True)

    #test.drop(labels = ["ID","v22","v107","v71","v31","v100","v63","v64"], axis = 1, inplace = True)
    test.drop(labels = ["ID","v22",'v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1, inplace = True)

    #test.drop(labels = ["ID","v22"], axis =1 , inplace = True)

    #find categorical variables
    categoricalVariables = []
    for var in train.columns:
        vector=pd.concat([train[var],test[var]], axis=0)
        typ=str(train[var].dtype)
        if (typ=='object'):
            categoricalVariables.append(var)


    print ("Generating dummies...")
    train, test = getDummiesInplace(categoricalVariables, train, test)

    #Remove sparse columns
    cls = train.sum(axis=0)
    train = train.drop(train.columns[cls<10], axis=1)
    test = test.drop(test.columns[cls<10], axis=1)

    print ("Filling in missing values...")
    fillNANStrategy = -1
    #fillNANStrategy = "mean"
    train = pdFillNAN(train, fillNANStrategy)
    test = pdFillNAN(test, fillNANStrategy)


    print ("Scaling...")
    train, scaler = preprocess_data(train)
    test, scaler = preprocess_data(test, scaler)

    train = np.asarray(train, dtype=np.float32)        
    labels = np.asarray(labels, dtype=np.int32).reshape(-1,1)

    return (train , labels , test )



def getDataSet_3_2():

    class AdjustVariable(object):
        def __init__(self, name, start=0.03, stop=0.001):
            self.name = name
            self.start, self.stop = start, stop
            self.ls = None

        def __call__(self, nn, train_history):
            if self.ls is None:
                self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

            epoch = train_history[-1]['epoch']
            new_value = np.float32(self.ls[epoch - 1])
            getattr(nn, self.name).set_value(new_value)


    def preprocess_data(X, scaler=None):
        if not scaler:
            scaler = StandardScaler()       
            scaler.fit(X)
        X = scaler.transform(X)
        return X, scaler


            
    def getDummiesInplace(columnList, train, test = None):
        #Takes in a list of column names and one or two pandas dataframes
        #One-hot encodes all indicated columns inplace
        columns = []
        
        if test is not None:
            df = pd.concat([train,test], axis= 0)
        else:
            df = train
            
        for columnName in df.columns:
            index = df.columns.get_loc(columnName)
            if columnName in columnList:
                dummies = pd.get_dummies(df.ix[:,index], prefix = columnName, prefix_sep = ".")
                columns.append(dummies)
            else:
                columns.append(df.ix[:,index])
        df = pd.concat(columns, axis = 1)
        
        if test is not None:
            train = df[:train.shape[0]]
            test = df[train.shape[0]:]
            return train, test
        else:
            train = df
            return train
            
    def pdFillNAN(df, strategy = "mean"):
        #Fills empty values with either the mean value of each feature, or an indicated number
        if strategy == "mean":
            return df.fillna(df.mean())
        elif type(strategy) == int:
            return df.fillna(strategy)



    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')

    np.random.seed(3210)
    train = train.iloc[np.random.permutation(len(train))]

    #Drop target, ID, and v22(due to too many levels), and high correlated columns
    labels = train["target"]
    trainId = train["ID"]
    testId = test["ID"]

    #train.drop(labels = ["ID","target","v22","v107","v71","v31","v100","v63","v64"], axis = 1, inplace = True)
    #train.drop(['ID','target',"v22",'v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1, inplace = True)

    train.drop(['ID','target',"v22"],axis=1, inplace = True)

    #test.drop(labels = ["ID","v22","v107","v71","v31","v100","v63","v64"], axis = 1, inplace = True)
    #test.drop(labels = ["ID","v22",'v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1, inplace = True)

    test.drop(labels = ["ID","v22"], axis =1 , inplace = True)

    #find categorical variables
    categoricalVariables = []
    for var in train.columns:
        vector=pd.concat([train[var],test[var]], axis=0)
        typ=str(train[var].dtype)
        if (typ=='object'):
            categoricalVariables.append(var)


    print ("Generating dummies...")
    train, test = getDummiesInplace(categoricalVariables, train, test)

    #Remove sparse columns
    cls = train.sum(axis=0)
    train = train.drop(train.columns[cls<10], axis=1)
    test = test.drop(test.columns[cls<10], axis=1)

    print ("Filling in missing values...")
    fillNANStrategy = -1
    #fillNANStrategy = "mean"
    train = pdFillNAN(train, fillNANStrategy)
    test = pdFillNAN(test, fillNANStrategy)


    print ("Scaling...")
    train, scaler = preprocess_data(train)
    test, scaler = preprocess_data(test, scaler)


    train = np.asarray(train, dtype=np.float32)        
    labels = np.asarray(labels, dtype=np.int32).reshape(-1,1)







"""
 Way to combine multiple models and multiple data sets 
 enumerate over the data sets and ensure that the index sychrnozies between them. 


the models will be stored in model/..


"""

############# MODELS ####################

# create the model0 working on dataset in model 0 

model0__name = [
        'xgboost_150_500_10',
        'xgboost_40',
        'xgboost_15',
        'xgboost_100_500_30',
        'xgboost_40_50',
        'xgboost_40_80',
        'xgboost_50_150',
        'xgboost_60_200',
        'xgboost_80_400'
        ]

model1__name = [
        'extraTree__fea_ext_ka2_750.pkl',
        'extraTree__fea_ext_ka2_1200.pkl',
        ]

model_2__name =  [
    'adaBoost__no_fea_ext_300.pkl' , 
    'adaBoost__no_fea_ext_400.pkl' , 
    'adaBoost__no_fea_ext_200.pkl' , 
    'gradientBoost__fea_ext_100.pkl',
    'gradientBoost__fea_ext_300.pkl',

]

model_3_2__name = [
    'neural_net.pkl',

]
