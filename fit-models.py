### TODO
# hyper-parameter optimization using bayesian optimization & gaussian process from scikit- optimize

"""
Fit the model on the training set, test on validation set
"""
import dataset
import models
from keras.callbacks import EarlyStopping, ModelCheckpoint

CNN_CHECKPOINT = 'cnn_checkpoint.hdf5'
RNN_CHECKPOINT = 'rnn_checkpoint.hdf5'
RNN_M_CHECKPOINT = 'rnn_m_checkpoint.hdf5' # lazy, just changed in model by hand.

def fit(model, batch_size, checkpoint_path):

    # define callbacks
    cb_earlystop = EarlyStopping(
            monitor = 'val_loss',
            mode = 'min',
            patience = 2,
            min_delta = .001,
            #save_best_only=True
            )
    
    cb_checkpoint = ModelCheckpoint(
            checkpoint_path, 
            monitor = 'val_loss', 
            mode = 'min',
            verbose = 1,
            save_best_only = True, 
            )
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])   

    model.fit(data['X_train'], data['y_train'], 
              epochs = 20, 
              batch_size = batch_size,  
              callbacks = [cb_earlystop, cb_checkpoint],
              validation_data = (data['X_valid'], data['y_valid']),
              shuffle = True
            )
    

    
# load the model_params and data dictionaries (see data.py)
data, model_params = dataset.load(validation_size = 0.1, use_glove = True)#, force_rebuild = True)

# build the model
#cnn_model = models.build_cnn_model(model_params, data)
#lstm_model = models.build_lstm_model(model_params, data)
mlstm_model = models.build_mlstm_model(model_params, data)

# fit the model

#fit(model = cnn_model, batch_size = 128, checkpoint_path = CNN_CHECKPOINT)
#fit(model = lstm_model, batch_size = 128, checkpoint_path = RNN_CHECKPOINT)
fit(model = mlstm_model, batch_size = 128, checkpoint_path = RNN_M_CHECKPOINT)

'''
import pandas as pd
test = pd.read_csv('../input/test.csv')

y_preds_cnn = cnn_model.predict(data['X_test'], batch_size=128)
y_preds_rnn = rnn_model.predict(data['X_test'], batch_size=128)    

y_preds = (y_preds_cnn + y_preds_rnn) / 2

submit = pd.DataFrame()
    
submit['id'] = test.loc[:, 'id']
    
submit = pd.concat([submit, pd.DataFrame(y_preds, columns=data['y_cols'])], axis=1)
submit.to_csv('submission.csv', index=False)
'''