"""

trying to use bnp paribas data set 
    and lasagne 

'''
    X_train = X_train.get_value()
    X_valid = X_valid.get_value()
    y_valid = y_valid.get_value()
    y_train = y_train.get_value()
'''
"""

import bnp 
import theano 
import theano.tensor as T 
import lasagne as ln 
import lasagne
import numpy as np 

def batch_gen(X , y , N): 
    while True:
        idx = np.random.choice(len(y),N)
        yield X[idx].astype('float32') , y[idx].astype('int32')


def batch_gen_valid(X , y , N): 
    while True:
        idx = np.random.choice(1000,N)
        yield X[idx].astype('float32') , y[idx].astype('int32')



def ln_test():

    n_train = 10000;
    print("Reading data .... ")
    data = bnp.readDataRS_bnp(n_train)

    train = data[0]
    valid = data[1]

#    for i in range(100):
#        print train[0][i] , train[1][i]

    print(train)


    X_train , y_train = train 
    X_valid , y_valid = valid 

    print("Completed Reading data .... ")

    # input layer , takes in 131 inputs 
    l_in = ln.layers.InputLayer((None,131))
    # hidden layer , takes in 100 inputs 
    l_hid_1 = ln.layers.DenseLayer(l_in , num_units = 150 , nonlinearity = ln.nonlinearities.softmax)
   # l_hid_2 = ln.layers.DenseLayer(l_hid_1 , num_units = 240 , nonlinearity = ln.nonlinearities.softmax)
   # l_hid_4 = ln.layers.DenseLayer(l_hid_2 , num_units = 150 , nonlinearity = ln.nonlinearities.softmax)
   # l_hid_7 = ln.layers.DenseLayer(l_hid_4 , num_units = 10 , nonlinearity = ln.nonlinearities.softmax)

    l_out = ln.layers.DenseLayer(l_hid_1 ,num_units = 2,nonlinearity = ln.nonlinearities.softmax)

    X_sym  = T.matrix()
    Y_sym = T.ivector()

    output = ln.layers.get_output(l_out , X_sym)
    pred = output.argmax(-1)

    loss = T.mean(ln.objectives.categorical_crossentropy(output, Y_sym))
    acc = T.mean(T.eq(pred , Y_sym))
    params = ln.layers.get_all_params(l_out)

    grad = T.grad(loss,params)
    updates = ln.updates.adam(grad , params , learning_rate = 0.1)

    f_train = theano.function([X_sym, Y_sym], [loss , acc ], updates = updates)
    f_val = theano.function([X_sym, Y_sym],  [loss , acc ])
    f_predict = theano.function([X_sym] , pred)
    ### TESTING FUNCTIONS 

    BATCH_SIZE = 20
    N_BATCHES = len(y_train) // BATCH_SIZE
    N_VAL_BATCHES = len(y_valid) // BATCH_SIZE

    print("X_train")
    print(type(X_train));
    print("X_valid")  
    print (type(X_valid))
    print("y_valid") 
    print (type(X_train))
    print("y_train" )
    print( type(X_train))

    train_batches = batch_gen(X_train , y_train , BATCH_SIZE)
    val_batches = batch_gen_valid(X_valid , y_valid , BATCH_SIZE) 
 
	
    print(" Training .... ")
    for epoch in range(n_train/BATCH_SIZE):
        train_loss = 0
        train_acc = 0 
        for _ in range(N_BATCHES):
            X , y = next(train_batches)
            loss , acc = f_train(X,y)
            train_loss += loss
            train_acc += acc  
        train_loss /= N_BATCHES
        train_acc /= N_BATCHES
        
        val_loss = 0
        val_acc = 0 
        
        for _ in range(N_VAL_BATCHES):
            X , y = next(val_batches)
            loss , acc = f_val(X,y)
            val_loss += loss 
            val_acc += acc
        val_loss /= N_VAL_BATCHES
        val_acc /= N_VAL_BATCHES
        
        print('train loss : ' , train_loss)
        print('train acc : ' , train_acc)
        print('val loss : ' , val_loss)
        print('val acc : ' , val_acc)


    print( " Applying model on test set ");
    predict_model = theano.function([X_sym] , (pred , output));
	
    datasets = bnp.readTest_N_bnp(10000)
    test_set_x = datasets[1];
    
  #  test_set_x = test_set_x.get_value()
   
    print(test_set_x)

    predicted_values = predict_model(test_set_x)

    print("dataset ...")
    print(test_set_x)
    
    #for i in range(100):
    #    print test_set_x[i];

    print("predicted values ...")
    print(predicted_values)
 
    # Writing to File 
    output = open('submission.csv','w');
    output.write('ID,PredictedProb\n');
    test_id = 0 ;

    print ("Writing to file")
    for i in range(predicted_values[0].shape[0]):
        print (predicted_values[0][i] , predicted_values[1][i] )
        print("Input ..")
        #print(test_set_x[i])
        ans = predicted_values[1][i][1]; # 1 or 0 from the result of the traniing
        output_line = str(datasets[0][test_id]) + ',' + str(ans) + '\n'
        output.write(output_line); 
        test_id += 1 ; 
    
    output.close()


if __name__ == '__main__' :
    ln_test()



