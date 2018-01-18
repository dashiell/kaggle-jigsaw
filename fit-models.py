### TODO
# hyper-parameter optimization using bayesian optimization & gaussian process from scikit- optimize

"""
Fit the model on the training set, test on validation set
"""
import dataset
import models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import gc


CNN_CHECKPOINT = 'cnn_checkpoint.hdf5'
RNN_CHECKPOINT = 'rnn_checkpoint.hdf5'
RNN_M_CHECKPOINT = 'rnn_m_checkpoint.hdf5' # lazy, just changed in model by hand.

PATH_YH_TRAIN_ALL = 'yh_train_all.npy'
PATH_YH_VAL_ALL = 'yh_train_all.npy'
PATH_YH_TEST_ALL = 'yh_test_all.npy'

def fit(model, batch_size, checkpoint_path):

    # define callbacks
    cb_earlystop = EarlyStopping(
            monitor = 'val_loss',
            mode = 'min',
            patience = 3,
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
    #import numpy as np
    #data['X_train']['has_utc'] = np.array(data['X_train']['has_utc'])

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
cnn_model = models.build_cnn_model(model_params, data)
lstm_model = models.build_lstm_model(model_params, data)
mlstm_model = models.build_mlstm_model(model_params, data)

# fit the model

#.431 (1,20)
#fit(model = cnn_model, batch_size = 128, checkpoint_path = CNN_CHECKPOINT)
#fit(model = lstm_model, batch_size = 128, checkpoint_path = RNN_CHECKPOINT)
#fit(model = mlstm_model, batch_size = 128, checkpoint_path = RNN_M_CHECKPOINT)

### TODO.. .this should be built on the oof predictions of k-fold CV fitted models.  But....
# that will take forever to train.
# at least rebuild the validation set before fitting each model next time, to approximate this.

cnn_model.load_weights(CNN_CHECKPOINT)
cnn_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 
y_tr_cnn = cnn_model.predict(data['X_train'], batch_size = 128)
y_v_cnn =  cnn_model.predict(data['X_valid'], batch_size = 128)
y_pred_cnn = cnn_model.predict(data['X_test'], batch_size = 128)
del cnn_model; gc.collect()
print('done')
lstm_model.load_weights(RNN_CHECKPOINT)
lstm_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 
y_tr_lstm = lstm_model.predict(data['X_train'], batch_size = 128)
y_v_lstm =  lstm_model.predict(data['X_valid'], batch_size = 128)
y_pred_lstm = lstm_model.predict(data['X_test'], batch_size = 128)
del lstm_model; gc.collect()
print('done')
mlstm_model.load_weights(RNN_M_CHECKPOINT)
mlstm_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy']) 
y_tr_mlstm = mlstm_model.predict(data['X_train'], batch_size = 128)
y_v_mlstm = mlstm_model.predict(data['X_valid'], batch_size = 128)
y_pred_mlstm = mlstm_model.predict(data['X_test'], batch_size = 128)
del mlstm_model; gc.collect()
print('done')


y_tr_all = np.hstack((y_tr_cnn, y_tr_lstm, y_tr_mlstm))
y_v_all = np.hstack((y_v_cnn, y_v_lstm, y_v_mlstm))
y_te_all = np.hstack((y_pred_cnn, y_pred_lstm, y_pred_mlstm))

np.save(PATH_YH_TRAIN_ALL, y_tr_all)
np.save(PATH_YH_VAL_ALL, y_v_all)
np.save(PATH_YH_TEST_ALL, y_te_all)


'''
ensemble = models.build_ensemble(y_preds_all, data)
ensemble.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
ensemble.fit(data['X_train'], data['y_train'], 
          epochs = 20, 
          batch_size = 128,  
          #callbacks = [cb_earlystop, cb_checkpoint],
          validation_data = (data['X_valid'], data['y_valid']),
          shuffle = True
        )


import pandas as pd
test = pd.read_csv('../input/test.csv')


y_preds = (y_pred_cnn + y_pred_lstm + y_pred_mlstm) / 3

submit = pd.DataFrame()
    
submit['id'] = test.loc[:, 'id']

### todo, clipping
## todo, if common obsenity present, at least mark 1 flag 1, if this holds true

submit = pd.concat([submit, pd.DataFrame(y_preds, columns=data['y_cols'])], axis=1)
submit.to_csv('submission.csv', index=False)
'''