'''
Find the best hyperparameters for our GRU model using scikit's bayesian optimization
'''

import dataset
#from model import models
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
import pandas as pd
import train_utils
import numpy as np


####

from keras.layers import Input, Embedding, Conv1D, Conv2D, \
    MaxPooling1D, Dense ,Dropout, SpatialDropout1D, LSTM, GRU, Bidirectional, \
    CuDNNGRU, CuDNNLSTM, Activation, Flatten, InputLayer, BatchNormalization, \
    concatenate, merge, Reshape, Lambda, \
    GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D, \
    MaxPooling1D, AveragePooling1D
from keras.layers.advanced_activations import PReLU, ELU
from keras.utils import multi_gpu_model
from keras.models import Model, Sequential
import tensorflow as tf
####

import gc
from keras import backend as K



'''

def get_dimensions():
    
    dim_batch_size = Integer(low = 64, high = 512, name='batch_size') 
    dim_n_dense_layers = Integer(low = 1, high = 3, name = 'n_dense_layers')
    dim_n_dense_outputs = Integer(low = 16, high = 128, name = 'n_dense_outputs')
    dim_dropout_rate = Real(low = 0.01, high = 0.7, name = 'dropout_rate')
    dim_l2_rate = Real(low = 1e-15, high = 1e-2, prior = 'log-uniform', name='l2_rate')
    
    dimensions = [
                  dim_batch_size,
                  dim_n_dense_layers,
                  dim_n_dense_outputs,
                  dim_dropout_rate,
                  dim_l2_rate,
                  ]

    return dimensions


@use_named_args(dimensions = get_dimensions())
def fitness(batch_size, n_dense_layers, n_dense_outputs, 
                dropout_rate, l2_rate):
       
    x1,x2,y1,y2 = dataset.get_train_valid(X_train, dataset.y_train, test_size=0.1)
    
    model = models.lstm_model(EMB_MATRIX, WORD_SEQ_LEN)
    #m.build_meta(n_dense_layers, n_dense_outputs, dropout_rate, l2_rate)
    model = train_utils.fit_on_val(model, 128, x1,x2,y1,y2)
    y_pred = model.predict(x2)
    validation_loss = train_utils.calc_loss(y2, y_pred)
    print('val loss', validation_loss)
    del model; gc.collect()
    K.clear_session()
    
    return validation_loss
'''
X_train, X_test, embedding_matrix, missing_words = dataset.get_keras_word_data()

#X_train, X_test, EMB_MATRIX = dataset.get_fasttext()    

x1,x2,y1,y2 = dataset.get_train_valid(X_train,dataset.y_train,0.05)

#EMB_MATRIX = dataset.get_glove()
WORD_SEQ_LEN = dataset.WORD_SEQ_LEN




#default_params = [128, 2, 100, 0.4, 5.80236e-05]
#fitness(default_params)
'''
search_results = gp_minimize(func = fitness,
                             dimensions = get_dimensions(),
                             acq_func='EI', # expected improvement
                             n_calls = 11,
                             #x0=default_params
                             )
                         
plot_convergence(search_results)
'''
'''
best_params = search_results.x

print("best_params: ", best_params)

# get the best 5 fits. and make an ensembled meta dataset from them.
all_fits = sorted(zip(search_results.func_vals, search_results.x_iters))

fitdata = {'loss' : [i[0] for i in all_fits],
        'batch_size' : [i[1][0] for i in all_fits],
        'params' : [i[1][1:] for i in all_fits]
}

pd.DataFrame(fitdata).to_csv('meta-model-results-new.csv', index=False)

'''
# emb_weights = model.layers[1].get_weights()[0]
# mask = [np.sum(emb_matrix,axis=1) == 0]
# emb_matrix[mask] = emb_weights[mask]
# emb_matrix[0] = np.zeros((300,))

#9853, one bidir, 64 units, top_k = 3, 5 epochs

def gru_model(embedding_mat, seq_len):
    
    inp = Input(shape = (seq_len, ) )
    
    x = Embedding(input_dim = embedding_mat.shape[0], #vocab size
                output_dim = embedding_mat.shape[1],
                weights = [embedding_mat],
                input_length = seq_len,
                trainable = False)(inp)
    
    # The benefit of adding SpatialDropout over normal keras dropout is that  
    # SpatialDropout entire embedding channels are dropped while the normal 
    # Keras embedding dropout drops all channels for entire words, and 
    # sometimes losing one or more words can alter the meaning completely.

    rnn_units = 64
    
    x = SpatialDropout1D(0.5)(x)
    gru = Bidirectional( CuDNNGRU(rnn_units, return_sequences=True) ) (x) 
    lstm = Bidirectional( CuDNNLSTM(rnn_units, return_sequences=True) ) (x)
    #x = Bidirectional( CuDNNGRU(64, return_sequences=True) )(x)    
    #x = Conv1D(128, kernel_size = 3, strides = 1, padding='same') (x)        
    #x = BatchNormalization() (x)
    #x = PReLU() (x)
    
    gru_max = MaxPooling1D(pool_size = 3, strides = 2, padding='same') (gru) 
    lstm_max = GlobalMaxPooling1D() (lstm)
    lstm_avg = GlobalAveragePooling1D() (lstm)

    top_k = 3
    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        # GlobalMaxpooling1d will output a vec of rnn_units (*2 for bidirectional)
        # We are getting the top 3
        return tf.reshape(k_max[0], (-1, rnn_units * 2 * top_k)) # max filter size =512 // 128
    
    # Transform the 512 x k features into a single vector
    gru_top_k = Lambda(_top_k, output_shape=(rnn_units * 2 * top_k,))(gru_max) # 128 * top_k
    
    x = concatenate([gru_top_k, lstm_max, lstm_avg])
    
    x = Dense(6, activation='sigmoid')(x) 

    model = Model(inputs = [inp], outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

#model = models.fasttext_model(np.max(X_train) +1)
model = gru_model(embedding_matrix, WORD_SEQ_LEN)
model = train_utils.fit_on_val(model, 64, x1,x2,y1,y2)
#model.fit(x=X_train,y=dataset.y_train, batch_size=64, epochs=5)

y_preds = model.predict(X_test, batch_size=128)

submit = pd.DataFrame()

submit['id'] = dataset.test.loc[:, 'id']

submit = pd.concat([submit, pd.DataFrame(y_preds, columns=dataset.y_cols)], axis=1)
submit.to_csv('../output/subm.csv', index=False)

