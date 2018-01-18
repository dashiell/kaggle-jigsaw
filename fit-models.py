### TODO
# hyper-parameter optimization using bayesian optimization & gaussian process from scikit- optimize
# stop passing around embed_dict; make dataset a singleton

"""
Fit the model on the training set, test on validation set
"""
import dataset
import models
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
import numpy as np
import gc


CNN_CHECKPOINT = 'cnn_checkpoint.hdf5'
RNN_CHECKPOINT = 'rnn_checkpoint.hdf5'
RNN_M_CHECKPOINT = 'rnn_m_checkpoint.hdf5' # lazy, just changed in model by hand.

PATH_YH_TRAIN_ALL = 'yh_train_all.npy'
PATH_YH_VAL_ALL = 'yh_train_all.npy'
PATH_YH_TEST_ALL = 'yh_test_all.npy'

def fit(x1, x2, y1, y2, model, batch_size, checkpoint_path):

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
    #import numpy as np
    #data['X_train']['has_utc'] = np.array(data['X_train']['has_utc'])

    model.fit(x1, y1, 
              epochs = 20, 
              batch_size = batch_size,  
              callbacks = [cb_earlystop, cb_checkpoint],
              validation_data = (x2, y2),
              shuffle = False
            )
    

    
# load the embed_dict and data dictionaries (see data.py)
X_train, X_test, y_train, embed_dict = dataset.load()

# build the model

#lstm_model = models.build_lstm_model(embed_dict)
#mlstm_model = models.build_mlstm_model(embed_dict)

# predict on oof and test
n_splits = 10

skf = StratifiedKFold(n_splits=n_splits, shuffle=True )

any_positive_cat = np.sum(y_train, axis = 1)

# the out of fold predictions
oof_preds = np.zeros(shape=y_train.shape)

# full test set predictions
test_preds = []
for i in np.arange(0, n_splits):
    test_preds.append( np.zeros(shape=(X_test.shape[0], 6)) )
    

fold = 0

X_test_keras = dataset.get_keras_dict(embed_dict, X_test)

for train_index, valid_index in skf.split(X_train, any_positive_cat):
    cnn_model = models.build_cnn_model(embed_dict)
    x1, x2 = X_train[train_index], X_train[valid_index]
    y1, y2 = y_train[train_index], y_train[valid_index]
    
    # X_train, X_valid for keras
    x1k = dataset.get_keras_dict(embed_dict, x1)
    x2k = dataset.get_keras_dict(embed_dict, x2)
    
    
    fit(x1k,x2k,y1,y2, model = cnn_model, batch_size = 128, checkpoint_path = CNN_CHECKPOINT)
    cnn_model.load_weights(CNN_CHECKPOINT)
    cnn_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    # create the out-of-fold predictions 
    oof_preds[valid_index] = cnn_model.predict(x2k, batch_size = 128)
    # create the predictions on the test set
    test_preds[fold] = cnn_model.predict(X_test_keras, batch_size = 128)
    
    del cnn_model
    gc.collect()

# get mean of all the test_pred matrices
test_mean = np.mean(test_preds, axis=0)

'''

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