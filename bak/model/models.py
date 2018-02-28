import keras
import tensorflow as tf
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D, \
    MaxPooling1D, Dense ,Dropout, LSTM, GRU, Bidirectional, CuDNNGRU, \
    CuDNNLSTM, Activation, Flatten, InputLayer, BatchNormalization, \
    concatenate, Lambda
from keras.layers.advanced_activations import PReLU
from keras.models import Model

#from keras.optimizers import RMSprop
from model.multiplicative_lstm import MultiplicativeLSTM
from keras.initializers import RandomNormal
from keras import backend as K
from model.keras_loss import auc_roc

from keras.utils import plot_model

def gru_model(embedding_mat, seq_len):
    
    inp = Input(shape = (seq_len, ) )
    
    x = Embedding(input_dim = embedding_mat.shape[0], #vocab size
                output_dim = embedding_mat.shape[1],
                weights = [embedding_mat],
                input_length = seq_len,
                trainable = False)(inp)
    # CuDNNGRU 
    x = Bidirectional( CuDNNGRU(64, return_sequences=True) )(x) 
    x = Bidirectional(CuDNNGRU(64, return_sequences=False))(x)

    x = Dropout(0.5)(x)
    x = Dense(64, activation='elu')(x)
    
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
    x = Bidirectional( CuDNNLSTM(100, return_sequences=True) )(x) #(emb)
    x = Bidirectional( CuDNNLSTM(100, return_sequences=False) )(x) 
    
    x = Dense(128, activation='elu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(64)(x)
    
    x = Dense(6, activation='sigmoid')(x) 
    
    model = Model(inputs = [inp], outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model
    
def cnn_model(embedding_mat, seq_len):
    inp = Input(shape = (seq_len, ) )
    
    x = Embedding(input_dim = embedding_mat.shape[0], #vocab size
                output_dim = embedding_mat.shape[1],
                input_length = seq_len,
                weights = [embedding_mat],
                trainable = False
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
    
    model = Model(inputs = [inp], outputs = x)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

#embed 16
def vdcnn_model(embedding_size=97, seq_maxlen=512, n_quantized_characters=97):
    """Very Deep CNN model, uses encoded character level data
    https://arxiv.org/pdf/1606.01781.pdf"""
       
    input_text = Input(shape=(seq_maxlen,))
    x = Embedding(input_dim=n_quantized_characters, output_dim=embedding_size,
                  input_length=seq_maxlen)(input_text)
    #x = _convolutional_block(64)(x)
    x = Conv1D(filters=64, kernel_size=3, strides=2, padding="same")(x)

    #filters = [64, 128, 256, 512]
    filters = [64, 128, 256, 512, 1024]

    for n_filt in filters:
        ### convolutional block
        x = Conv1D(n_filt, kernel_size=3, strides=1, padding='same', 
                   activation='linear',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.001)) (x)
    
        x = BatchNormalization() (x)
        x = PReLU() (x) #PReLU
        
        x = Conv1D(n_filt, kernel_size=3, strides=1, padding='same', 
                   activation='linear',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.001)) (x)
    
        x = BatchNormalization() (x)
        x = PReLU() (x)
        ###
        #if n_filt != filters[-1]:
        #    print("adding", n_filt, filters[-1])
        x = MaxPooling1D(pool_size=3, strides=2, padding='same') (x)

    # k-max pooling (Finds values and indices of the k largest entries for the last dimension)
    top_k = 8
    def _top_k(x):
        x = tf.transpose(x, [0, 2, 1])
        k_max = tf.nn.top_k(x, k=top_k)
        return tf.reshape(k_max[0], (-1, filters[-1] * top_k))
    
    k_max = Lambda(_top_k, output_shape=(filters[-1] * top_k,))(x)


    #x = GlobalMaxPool1D()(x)
    
    x = Dropout(0.2)(Dense(128)(k_max))
    x = PReLU() (x)
    x = Dropout(0.2)(Dense(128)(x))
    x = PReLU() (x)
        

    y_pred = Dense(6, activation='sigmoid')(x)
    model = Model(inputs=input_text, outputs=y_pred)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', auc_roc])
    
    return model