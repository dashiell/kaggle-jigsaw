import keras
import tensorflow as tf
from keras.layers import Input, Embedding, Conv1D, \
    MaxPooling1D, Dense ,Dropout, SpatialDropout1D, LSTM, GRU, Bidirectional, \
    CuDNNGRU, CuDNNLSTM, Activation, Flatten, InputLayer, BatchNormalization, \
    concatenate, merge, Reshape, Lambda, \
    GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D, \
    MaxPooling1D, AveragePooling1D
from keras.layers.advanced_activations import PReLU, ELU
from keras.utils import multi_gpu_model

from keras.models import Model, Sequential
from keras.optimizers import SGD

#from keras.optimizers import RMSprop
from model.multiplicative_lstm import MultiplicativeLSTM
from keras.initializers import RandomNormal
from keras import backend as K
from model.keras_loss import auc_roc

from keras.utils import plot_model

def meta_model(x_meta_dim):
    X_meta_inp = Input(shape = (x_meta_dim, ), name = 'X_meta')
    
    x = Dense(128, activation='relu') (X_meta_inp)
    
    x = Dense(64, activation='relu') (X_meta_inp)
    
    #x = Dense(64, activation='relu') (x)
    
    out = Dense(6, activation='sigmoid') (x)
    
    with tf.device('/cpu:0'):
        model = Model(inputs = [X_meta_inp], 
                      outputs = out)
    # plot_model(model, show_shapes=True)
    model = multi_gpu_model(model, gpus=2)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

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
    
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional( CuDNNGRU(64, return_sequences=True) )(x) 
    x = Bidirectional( CuDNNGRU(64, return_sequences=True) )(x)
    
    avg_gru = GlobalAveragePooling1D() (x)
    max_gru = GlobalMaxPooling1D() (x)
    max_gru2 = GlobalMaxPooling1D(pool_size = 3, strides = 2, padding='same') (x)
      
    x = concatenate([avg_gru, max_gru, max_gru2])
    
    x = Dense(6, activation='sigmoid')(x) 
    
    model = Model(inputs = [inp], outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

def lstm_model(embedding_mat, seq_len):
    
    inp = Input(shape = (seq_len, ) )
    
    x = Embedding(input_dim = embedding_mat.shape[0], #vocab size
                output_dim = embedding_mat.shape[1],
                weights = [embedding_mat],
                input_length = seq_len,
                trainable = False) (inp)
    # CuDNNLSTM
    x = Bidirectional( LSTM(100, return_sequences=True, dropout=0.2, recurrent_dropout=0.1) )(x) #(emb)
    x = Bidirectional( LSTM(100, return_sequences=False, recurrent_dropout=0.1) )(x) 
    
    x = Dense(128, activation='elu')(x)
    x = Dropout(0.1)(x)
    
    x = Dense(64)(x)
    
    x = Dense(6, activation='sigmoid')(x) 
    #with tf.device('/cpu:0'):
    model = Model(inputs = [inp], outputs = x)
    #model = multi_gpu_model(model, gpus=2)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

def fasttext_model(input_dim):
    model = Sequential()
    model.add(Embedding(input_dim=input_dim, output_dim=20))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(6, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model
    
def cnn_model(embedding_mat, seq_len):
    inp = Input(shape = (seq_len, ) )
    
    x = Embedding(input_dim = embedding_mat.shape[0], #vocab size
                output_dim = embedding_mat.shape[1],
                input_length = seq_len,
                weights = [embedding_mat],
                trainable = True
                ) (inp)
    
    prefilt_x = Dropout(0.5) (x)
    
    # our convolutional layers that we build in the below for loop
    convolutions = []
    
    # 3 convolutions with increasing filter length, dilation rate
    for i in range(1,4):
        x = prefilt_x
        
        x = Conv1D(filters = 100,
                   kernel_size = i, # filter length
                   padding = 'same', 
                   activation = 'elu', 
                   dilation_rate = i, # captures ngrams 1 (default),2,3
                   strides = 1) (x) # strides must == 1 when dilation_rate != 1
    
        # Pooling layer; subsample the result of each filter. 
        # Gives a single real number each filter.
    
        x = Dropout(0.25)(GlobalMaxPool1D() (x)) 
        
        convolutions.append(x)
    
    # merge convs into a single tensor
    x = concatenate(convolutions)
    
    x = Dense(64, activation='elu') (x)
    x = Dropout(0.1) (x)
 
    x = Dense(6, activation='sigmoid')(x) 
    
    model = Model(inputs = [inp], outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

def mlstm_model(embedding_mat, seq_len):
    inp = Input(shape = (seq_len, ) )
    
    x = Embedding(input_dim = embedding_mat.shape[0], #vocab size
        output_dim = embedding_mat.shape[1],
        input_length = seq_len,
        weights = [embedding_mat],
        trainable = False
        ) (inp)

    x = Bidirectional(MultiplicativeLSTM(return_sequences = True, 
                                         units = 128, 
                                         dropout = 0.2, 
                                         recurrent_dropout = 0.2)) (x)
    
    x = GlobalMaxPool1D() (x)
    x = Dropout(0.5) (x)
    
    x = Dense(64, activation='elu')(x)
    
    x = Dense(6, activation='sigmoid')(x)
    #with tf.device('/cpu:0'):
    model = Model(inputs = [inp], outputs = x)
    #model = multi_gpu_model(model, gpus=2)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

#embed 16
def vdcnn_model( seq_len, vocab_size):
    """Very Deep CNN model, uses encoded character level data
    https://arxiv.org/pdf/1606.01781.pdf"""
       
    input_text = Input(shape = (seq_len,))
    x = Embedding(input_dim = vocab_size, 
                  output_dim = 16,
                  input_length = seq_len) (input_text)

    x = Conv1D(filters = 64, kernel_size = 3, strides = 2, padding="same")(x)

    filters = { 64 : 2, 128 : 2, 256 : 2, 512 : 2 }

    for n_filters, n_blocks in sorted(filters.items()):
        # Each block has two convlutions, batch norm, relu
        for i in range(n_blocks):
            ### convolutional block       
            x = Conv1D(n_filters, kernel_size = 3, strides = 1, padding='same') (x)        
            x = BatchNormalization() (x)
            x = PReLU() (x) 
            
            x = Conv1D(n_filters, kernel_size = 3, strides = 1, padding='same') (x)        
            x = BatchNormalization() (x)
            x = PReLU() (x) 
        
        x = MaxPooling1D(pool_size = 3, strides = 2, padding='same') (x)

    # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
    # these are the k most important features anywhere in the sentence.
    top_k = 8
    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, 512 * top_k)) # max filter size =512
    
    # Transform the 512 x k features into a single vector
    k_max = Lambda(_top_k, output_shape=(512 * top_k,))(x)

    #x = GlobalMaxPool1D()(x)
    
    x = Dropout(0.2)(Dense(512, kernel_initializer='he_normal')(k_max))
    x = PReLU() (x)
    x = Dropout(0.2)(Dense(512, kernel_initializer='he_normal')(x))
    x = PReLU() (x)
    y_pred = Dense(6, activation='sigmoid')(x)
    
    model = Model(inputs=input_text, outputs=y_pred)
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=False)

    model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy', auc_roc])
    
    return model