from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D, Dense 
from keras.layers import Dropout, LSTM, GRU, Bidirectional
from keras.layers import Activation, Flatten
from keras.layers.merge import concatenate
from keras.models import Model
from multiplicative_lstm import MultiplicativeLSTM
import keras

"""
Build the following models:
    Convolutional neural network
    Bidirectional LSTM
"""


'''
Keras CNN Model
'''
def build_cnn_model(embed_dict, n_filters = 100):
    
    # model input
    comment_text = Input(shape = (embed_dict['emb_input_seq_len'], ), name='comment_text')
    has_utc = Input(shape=[1], name='has_utc')
    #pct_caps = Input(shape=[1], name='pct_caps')
    #has_ethnicity = Input(shape=[1], name='has_ethnicity')
    #has_ipaddr = Input(shape=[1], name='has_ipaddr')
    
    # word embedding layer
    
    if embed_dict['use_glove']:
        print('using glove embeddings')
        emb_comment_text = Embedding(input_dim = embed_dict['emb_vocab_size'], 
                      output_dim = embed_dict['emb_out_size'],
                      weights = [embed_dict['embedding_matrix']],
                      input_length = embed_dict['emb_input_seq_len'],
                      trainable = False
                      ) (comment_text)
    else:
        emb_comment_text = Embedding(input_dim = embed_dict['emb_vocab_size'], 
                      output_dim = embed_dict['emb_out_size']) (comment_text)
    
    #try input_dim=1, output_dim=20
    emb_has_utc = Embedding(input_dim = 2, output_dim = 2, input_length = 1) (has_utc)
    
    
    # Regularization
    prefilt_x = Dropout(0.25) (emb_comment_text)
    
    # our convolutional layers that we build in the below for loop
    convolutions = []
    
    # 3 convolutions with increasing 
    for i in range(1,4):
        x = prefilt_x
        
        
        x = Conv1D(filters = n_filters,
                   kernel_size = i, # filter length
                   padding = 'same', 
                   activation = 'elu', 
                   dilation_rate = i, # captures ngrams 1 (default),2,3
                   strides=1) (x) # strides must == 1 when dilation_rate != 1
    
        # Pooling layer; subsample the result of each filter. Gives a single int each filter
        x = Dropout(0.25)(GlobalMaxPool1D() (x))
        
        convolutions.append(x)
    
    # merge convs into a single tensor
    convolutions = concatenate(convolutions)
    x = concatenate([
            Flatten()(emb_has_utc), #, has_ipaddr, 
            convolutions
            ])
    
    x = Dense(64, activation='elu') (x)
    x = Dropout(0.1) (x)
    x = Dense(32, activation='elu', kernel_regularizer=keras.regularizers.l2(0.01)) (x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation = 'sigmoid') (x)
    
    model = Model(inputs = [comment_text, has_utc], outputs = x)
    
    return model

# val_loss: 0.0530

'''
Keras Bidirectional RNN Model
'''

def build_lstm_model(embed_dict, n_units = 64):
    inp = Input(shape = (embed_dict['emb_input_seq_len'],), name='comment_text')
    
    if embed_dict['use_glove']:
        print('using glove embeddings')
        x = Embedding(input_dim = embed_dict['emb_vocab_size'], 
                      output_dim = embed_dict['emb_out_size'],
                      weights = [embed_dict['embedding_matrix']],
                      input_length = embed_dict['emb_input_seq_len'],
                      trainable = False
                      ) (inp)
    else:
        x = Embedding(input_dim = embed_dict['emb_vocab_size'], 
                      output_dim = embed_dict['emb_out_size']) (inp)
    
    lstm = LSTM(return_sequences = True, units = n_units)
    #mlstm = MultiplicativeLSTM(units = n_units, dropout = 0.2, recurrent_dropout = 0.2)
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
    x = Dense(6, activation = 'sigmoid') (x)
    
    model = Model(inputs = inp, outputs = x)

    return model

'''
Keras Bidirectional RNN Model
'''

def build_mlstm_model(embed_dict, n_units = 128):
    inp = Input(shape = (embed_dict['emb_input_seq_len'],), name='comment_text')
    
    if embed_dict['use_glove']:
        print('using glove embeddings')
        x = Embedding(input_dim = embed_dict['emb_vocab_size'], 
                      output_dim = embed_dict['emb_out_size'],
                      weights = [embed_dict['embedding_matrix']],
                      input_length = embed_dict['emb_input_seq_len'],
                      trainable = False
                      ) (inp)
    else:
        x = Embedding(input_dim = embed_dict['emb_vocab_size'], 
                      output_dim = embed_dict['emb_out_size']) (inp)
    

    # try with return_sequences = True and put back maxpooling
    mlstm = MultiplicativeLSTM(return_sequences = True, 
                               units = n_units, 
                               dropout = 0.2, 
                               recurrent_dropout = 0.2)

        
    x = Bidirectional(mlstm) (x)
    
    x = GlobalMaxPool1D() (x)
    x = Dropout(0.5) (x)
    x = Dense(64, activation = 'relu') (x)
    x = Dropout(0.1) (x)
    x = Dense(32, activation = 'relu') (x)
    x = Dropout(0.1) (x)
    x = Dense(6, activation = 'sigmoid') (x)
    
    model = Model(inputs = inp, outputs = x, name='cnn_model')
    
    return model
    
def build_ensemble(y_preds_all):
#    inp = Input()
    
    y_preds_all = Input(shape=(y_preds_all.shape[1], ), name='y_preds_all')
    has_utc = Input(shape=[1], name='has_utc')
    pct_caps = Input(shape=[1], name='pct_caps')
    #has_ethnicity = Input(shape=[1], name='has_ethnicity')
    #has_ipaddr = Input(shape=[1], name='has_ipaddr')
    
    x = keras.layers.concatenate([y_preds_all, has_utc, pct_caps])
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(6, activation='sigmoid', name='output')(x)
    
    model = Model(inputs = [y_preds_all, has_utc], outputs=out)
    
    return model
    
