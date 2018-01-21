import dataset
import keras
from keras.layers import Input, Embedding, Conv1D, GlobalMaxPool1D, Dense 
from keras.layers import Dropout, LSTM, GRU, Bidirectional
from keras.layers import Activation, Flatten
from keras.layers.merge import concatenate
from keras.models import Model
from multiplicative_lstm import MultiplicativeLSTM
from keras.callbacks import EarlyStopping, ModelCheckpoint

class ModelBase:
    
    checkpoint_path = 'temp-checkpoint.h5py'
    embed_dict = dataset.get_embed_dict()


    
    def fit(self, x1, x2, y1, y2, epochs, batch_size):
    
        callbacks = [
        EarlyStopping(monitor = 'val_loss', 
                      mode = 'min',
                      patience = 2,
                      min_delta = .001
                      ),
        ModelCheckpoint(filepath = ModelBase.checkpoint_path, 
                        monitor = 'val_loss', 
                        mode = 'min',
                        verbose = 1,
                        save_best_only = True
                        )
        ]  
        self.model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])   
        
        self.model.fit(x1, y1, 
          epochs = epochs, 
          batch_size = batch_size,  
          callbacks = callbacks,
          validation_data = (x2, y2),
          shuffle = False
        )
    
    # load the best model weights and predict
    def predict_using_best_weights(self, X, batch_size):
        
        self.model.load_weights(ModelBase.checkpoint_path)
        
        self.model.compile(loss = 'binary_crossentropy', 
                           optimizer = 'adam', 
                           metrics = ['accuracy'])

        
        return self.model.predict(X, batch_size)
    
''' 
Keras CNN Model

Override the base class fit and predict methods 
'''
class CNNModel(ModelBase):
    
    def __init__(self):
        self.model = self.build()

    def build(self):
        n_filters = 100
        
        # model input
        comment_text = Input(shape = (ModelBase.embed_dict['emb_input_seq_len'], ), name='comment_text')
        #has_utc = Input(shape=[1], name='has_utc')
        #pct_caps = Input(shape=[1], name='pct_caps')
        #has_ethnicity = Input(shape=[1], name='has_ethnicity')
        #has_ipaddr = Input(shape=[1], name='has_ipaddr')
        
        # word embedding layer
        
        emb_comment_text = Embedding(input_dim = ModelBase.embed_dict['emb_vocab_size'], 
                          output_dim = ModelBase.embed_dict['emb_out_size'],
                          weights = [ModelBase.embed_dict['embedding_matrix']],
                          input_length = ModelBase.embed_dict['emb_input_seq_len'],
                          trainable = False
                          ) (comment_text)
        
        #emb_has_utc = Embedding(input_dim = 1, output_dim = 2, input_length = 1) (has_utc)
        
        # Regularization
        prefilt_x = Dropout(0.5) (emb_comment_text)
        
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
        x = concatenate(convolutions)
        
        x = Dense(64, activation='elu') (x)
        x = Dropout(0.1) (x)
        x = Dense(32, activation='elu', kernel_regularizer=keras.regularizers.l2(0.01)) (x)
        x = Dropout(0.1)(x)
        x = Dense(6, activation = 'sigmoid') (x)
        
        model = Model(inputs = comment_text, outputs = x)
        return model

''' 
Keras LSTM Model
'''

class LSTMModel(ModelBase):
    
    def __init__(self):
        
        self.model = self.build()
    
    def build(self):
        
        n_units = 64
        
        inp = Input(shape = (self.embed_dict['emb_input_seq_len'],), name='comment_text')
        
        if self.embed_dict['use_glove']:
            x = Embedding(input_dim = self.embed_dict['emb_vocab_size'], 
                          output_dim = self.embed_dict['emb_out_size'],
                          weights = [self.embed_dict['embedding_matrix']],
                          input_length = self.embed_dict['emb_input_seq_len'],
                          trainable = False
                          ) (inp)
        else:
            x = Embedding(input_dim = self.embed_dict['emb_vocab_size'], 
                          output_dim = self.embed_dict['emb_out_size']) (inp)
        
        lstm = LSTM(return_sequences = True, units = n_units, unroll=True)
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
    Keras multiplicative LSTM Model
'''
class MLSTMModel(ModelBase):
    
    def __init__(self):
        
        self.model = self.build()
    
    def build(self):
        
        n_units = 128
        inp = Input(shape = (self.embed_dict['emb_input_seq_len'],), name='comment_text')

        x = Embedding(input_dim = self.embed_dict['emb_vocab_size'], 
                          output_dim = self.embed_dict['emb_out_size'],
                          weights = [self.embed_dict['embedding_matrix']],
                          input_length = self.embed_dict['emb_input_seq_len'],
                          trainable = False
                          ) (inp)
          
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

class EnsembleModel(ModelBase):
    
    def __init__(self):
        
        self.model = self.build()

        
    def build(self):
        
        oof_preds = Input(shape=(18,), name='oof_preds')
        #has_utc = Input(shape=[1], name='has_utc')
        #pct_caps = Input(shape=[1], name='pct_caps')
        #has_ethnicity = Input(shape=[1], name='has_ethnicity')
        #has_ipaddr = Input(shape=[1], name='has_ipaddr')
        
        #x = keras.layers.concatenate([oof_preds, has_utc])
        x = oof_preds
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu')(x)
        #x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        #x = Dropout(0.1)(x)
        x = Dense(64, activation='relu')(x)
        out = Dense(6, activation='sigmoid', name='output')(x)
        
        model = Model(inputs = [oof_preds], outputs=out)
        return model
    
#mA.fit(55)