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
        idx = np.random.choice(2000,N)
        yield X[idx].astype('float32') , y[idx].astype('int32')


def ln_test():
    data = bnp.readDataRS_bnp(20000)

    train = data[0]
    valid = data[1]
    X_train , y_train = train 
    X_valid , y_valid = valid 


    l_in = ln.layers.InputLayer((None,131))
    l_hid = ln.layers.DenseLayer(l_in , num_units = 300 , nonlinearity = ln.nonlinearities.softmax)

    l_hid_2 = ln.layers.DenseLayer(l_hid, num_units = 100 , nonlinearity = ln.nonlinearities.softmax)

    l_out = ln.layers.DenseLayer(
    l_hid_2 ,
    num_units = 2,
    nonlinearity = ln.nonlinearities.softmax)

    X_sym  = T.matrix()
    Y_sym = T.ivector()

    output = ln.layers.get_output(l_out , X_sym)
    pred = output.argmax(-1)

    loss = T.mean(ln.objectives.categorical_crossentropy(output, Y_sym))
    acc = T.mean(T.eq(pred , Y_sym))
    params = ln.layers.get_all_params(l_out)
    print params

    grad = T.grad(loss,params)
    updates = ln.updates.adam(grad , params , learning_rate = 0.05)
    print updates

    f_train = theano.function([X_sym, Y_sym], [loss , acc ], updates = updates)

    f_val = theano.function([X_sym, Y_sym],  [loss , acc ])


    BATCH_SIZE = 64
    N_BATCHES = len(y_train) // BATCH_SIZE
    N_VAL_BATCHES = len(y_valid) // BATCH_SIZE


    train_batches = batch_gen(X_train , y_train , BATCH_SIZE)
    val_batches = batch_gen(X_valid , y_valid , BATCH_SIZE) 
    X ,y = next(train_batches)
    #plt.imshow(X[0].reshape((28,28)), cmap='gray' )
    print(y[0])


    for epoch in range(100):
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
        
        


if __name__ == '__main__' :
    ln_test()



