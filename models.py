from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D, Dense 
from keras.layers import Dropout, LSTM, GRU, Bidirectional
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.models import Model
import keras

"""
Build the following models:
    Convolutional neural network
    Bidirectional LSTM
"""


'''
Keras CNN Model
'''
def build_cnn_model(model_params, data, filters = 100):
    
    # model input
    inp = Input(shape = (model_params['emb_input_seq_len'], ))
    
    # word embedding layer
    
    if model_params['use_glove']:
        print('using glove embeddings')
        x = Embedding(input_dim = model_params['emb_vocab_size'], 
                      output_dim = model_params['emb_out_size'],
                      weights = [data['embedding_matrix']],
                      input_length = model_params['emb_input_seq_len'],
                      trainable = False
                      ) (inp)
    else:
        x = Embedding(input_dim = model_params['emb_vocab_size'], 
                      output_dim = model_params['emb_out_size']) (inp)
    
    # Regularization
    prefilt_x = Dropout(0.25) (x)
    
    # our convolutional layers that we build in the below for loop
    convolutions = []
    
    # 3 convolutions with increasing 
    for i in range(1,4):
        x = prefilt_x
        
        
        x = Conv1D(filters = 100,
                   kernel_size = i, # filter length
                   padding = 'same', 
                   activation = 'elu', 
                   dilation_rate = i, # captures ngrams 1 (default),2,3
                   strides=1) (x) # strides must == 1 when dilation_rate != 1
    
        # Pooling layer; subsample the result of each filter. Gives a single int each filter
        x = Dropout(0.25)(GlobalMaxPool1D() (x))
        
        convolutions.append(x)
    
    # merge convs into a single tensor
    x = concatenate(convolutions)
    
    x = Dense(64, activation='elu') (x)
    x = Dropout(0.1) (x)
    x = Dense(32, activation='elu', kernel_regularizer=keras.regularizers.l2(0.01)) (x)
    x = Dropout(0.1)(x)
    x = Dense(data['y_train'].shape[1], activation = 'sigmoid') (x)
    
    model = Model(inputs = inp, outputs = x)
    
    return model

# val_loss: 0.0530

'''
Keras Bidirectional RNN Model
'''

def build_rnn_model(model_params, data, n_units = 64):
    inp = Input(shape = (model_params['emb_input_seq_len'],))
    
    if model_params['use_glove']:
        print('using glove embeddings')
        x = Embedding(input_dim = model_params['emb_vocab_size'], 
                      output_dim = model_params['emb_out_size'],
                      weights = [data['embedding_matrix']],
                      input_length = model_params['emb_input_seq_len'],
                      trainable = False
                      ) (inp)
    else:
        x = Embedding(input_dim = model_params['emb_vocab_size'], 
                      output_dim = model_params['emb_out_size']) (inp)
    
    lstm = LSTM(return_sequences = True, units = n_units)
    #gru = GRU(return_sequences = True, units = 64)
    
    x = Bidirectional(lstm) (x)
    #x = Activation('elu') (x)
    #x = Bidirectional(gru) (x)
    #x = Activation('elu') (x)
    
    x = GlobalMaxPool1D() (x)
    x = Dropout(0.25) (x)
    x = Dense(64, activation = 'relu') (x)
    x = Dropout(0.1) (x)
    x = Dense(32, activation = 'relu') (x)
    x = Dropout(0.1) (x)
    x = Dense(data['y_train'].shape[1], activation = 'sigmoid') (x)
    
    model = Model(inputs = inp, outputs = x)

    return model
    

