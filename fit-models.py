### TODO
# hyper-parameter optimization using bayesian optimization & gaussian process from scikit- optimize
# stop passing around embed_dict; make dataset a singleton

"""
Fit the model on the training set, test on validation set
"""
import dataset
from Models import BuildModels
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from keras import backend as K
import numpy as np
import gc

TRAIN_META_PATH = '../input/train-meta.npy'
TEST_META_PATH = '../input/test-meta.npy'

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
              epochs = 1, 
              batch_size = batch_size,  
              callbacks = [cb_earlystop, cb_checkpoint],
              validation_data = (x2, y2),
              shuffle = False
            )
    

    
# load the embed_dict and data dictionaries (see data.py)
X_train, X_test, y_train, embed_dict = dataset.load()

build_models = BuildModels(embed_dict)

# build the model

#lstm_model = models.build_lstm_model(embed_dict)
#mlstm_model = models.build_mlstm_model(embed_dict)

# predict on oof and test

all_models = [build_models.cnn, build_models.lstm, build_models.mlstm]

train_meta_preds = np.zeros((y_train.shape[0], y_train.shape[1]*len(all_models)))
test_meta_preds = np.zeros((X_test.shape[0], y_train.shape[1]*len(all_models)))

n_splits = 5

skf = StratifiedKFold(n_splits=n_splits, shuffle=True )

any_positive_cat = np.sum(y_train, axis = 1)

for curr_model in all_models:
    model_ix = 0
    
    # the out of fold predictions
    oof_preds = np.zeros(shape=y_train.shape)
    
    # full test set predictions
    test_preds = []
    for i in np.arange(0, n_splits):
        test_preds.append( np.zeros(shape=(X_test.shape[0], 6)) )
        
    fold = 0
    
    X_test_keras = dataset.get_keras_dict(embed_dict, X_test)
    
    for train_index, valid_index in skf.split(X_train, any_positive_cat):
        model = curr_model()
        x1, x2 = X_train[train_index], X_train[valid_index]
        y1, y2 = y_train[train_index], y_train[valid_index]
        
        # X_train, X_valid for keras
        x1k = dataset.get_keras_dict(embed_dict, x1)
        x2k = dataset.get_keras_dict(embed_dict, x2)
        
        checkpoint_path = "checkpoint-%s.hdf5" % (curr_model.__name__)
        fit(x1k,x2k,y1,y2, model = model, batch_size = 128, checkpoint_path = checkpoint_path)
        model.load_weights(checkpoint_path)
        model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
        
        # create the out-of-fold predictions 
        oof_preds[valid_index] = model.predict(x2k, batch_size = 128)
        # create the predictions on the test set
        test_preds[fold] = model.predict(X_test_keras, batch_size = 128)
        
        fold +=1
        
        # free up some system and GPU memory for the next fit
        del model
        gc.collect()
        K.clear_session()
    
    # get mean of all the test_pred matrices
    test_mean = np.mean(test_preds, axis=0)
    
    first_col = model_ix * 6
    last_col = first_col + 6
    
    # put the model oof predictions into a big matrix of all model preds
    train_meta_preds[:, first_col:last_col] = oof_preds
    # put the models test predictions into a matrix of all model preds
    test_meta_preds[:, first_col:last_col] = test_mean
    
    # save this for 
    np.save(TRAIN_META_PATH, train_meta_preds)
    np.save(TEST_META_PATH, test_meta_preds)
    
    model_ix += 1

'''

# fit the model


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