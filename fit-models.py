### TODO
# hyper-parameter optimization using bayesian optimization & gaussian process from scikit- optimize
# stop passing around embed_dict; make dataset a singleton

"""
Fit the model on the training set, test on validation set
"""
import dataset
from Models import CNNModel, LSTMModel, MLSTMModel, EnsembleModel
from sklearn.model_selection import StratifiedKFold, train_test_split
from keras import backend as K
import numpy as np
import pandas as pd
import gc
import xgboost as xgb


TRAIN_META_PATH = '../meta/train.npy'
TEST_META_PATH = '../meta/test.npy'


def create_meta_data():
  
    # load the embed_dict and data dictionaries (see data.py)
    X_train, X_test, y_train = dataset.load()
    
    all_models = [CNNModel, LSTMModel, MLSTMModel]
    
    train_meta_preds = np.zeros((y_train.shape[0], y_train.shape[1]*len(all_models)))
    test_meta_preds = np.zeros((X_test.shape[0], y_train.shape[1]*len(all_models)))
    
    n_splits = 10
    
    any_positive_cat = np.sum(y_train, axis = 1)
    
    X_test_keras = dataset.get_keras_dict(X_test)
    
    model_ix = 0
    for curr_model in all_models:
        
        # the out of fold predictions
        oof_preds = np.zeros(shape=y_train.shape)
        
        # full test set predictions
        test_preds = []
        for i in np.arange(0, n_splits):
            test_preds.append( np.zeros(shape=(X_test.shape[0], 6)) )
            
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True )
        
        fold = 0
        for train_index, valid_index in skf.split(X_train, any_positive_cat):
            print('~~training fold:', fold)
            model = curr_model()
            
            x1, x2 = X_train[train_index], X_train[valid_index]
            y1, y2 = y_train[train_index], y_train[valid_index]
            
            # X_train, X_valid for keras
            x1k = dataset.get_keras_dict(x1)
            x2k = dataset.get_keras_dict(x2)
                   
            ######### batch size WAS ~~~~ 128 ~~~ #########
             
            model.fit(x1k,x2k,y1,y2, epochs = 20, batch_size = 128)
    
            # create the out-of-fold predictions 
            oof_preds[valid_index] = model.predict_using_best_weights(x2k, batch_size = 128)
            # create the predictions on the test set
            test_preds[fold] = model.predict_using_best_weights(X_test_keras, batch_size = 128)
            
            fold +=1
            
            # reset the weights, free up some system and GPU memory for the next fit
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


def get_keras_meta(oof_preds):
    X = {
            'oof_preds' : oof_preds,
            # add other data
    }
    return X

def train_on_meta():
    X_train = np.load(TRAIN_META_PATH)
    print(X_train.shape)
    y_cols = dataset.y_cols
    y_train = pd.read_csv('../input/train.csv')[y_cols].values
    
    x1,x2,y1,y2 = train_test_split(X_train, y_train, test_size=0.1, shuffle=True)
    
    X_test = np.load(TEST_META_PATH)
    
    print(y1.shape)
    '''
    x1k = get_keras_meta(x1)
    x2k = get_keras_meta(x2)
    
    model = EnsembleModel()
    model.fit(x1k,x2k,y1,y2, epochs = 20, batch_size = 64)
    '''
    from xgboost import XGBRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import log_loss
    
    model = MultiOutputRegressor(XGBRegressor(objective='reg:linear'))
    #model.fit(x1, y1)
    #preds = model.predict(x2)
    #ll = log_loss(y2, preds)
    #print('ll {}'.format(ll))
    
    model.fit(X_train, y_train)
    y_preds = model.predict(X_test)
    
    return y_preds




create_meta_data()
#y_preds = train_on_meta()            


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
'''

test = pd.read_csv('../input/test.csv')
submit = pd.DataFrame()
    
submit['id'] = test.loc[:, 'id']

### todo, clipping
## todo, if common obsenity present, at least mark 1 flag 1, if this holds true

submit = pd.concat([submit, pd.DataFrame(y_preds, columns=dataset.y_cols)], axis=1)
submit.to_csv('submission.csv', index=False)