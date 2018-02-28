import numpy as np
from sklearn.metrics import log_loss, roc_auc_score
import gc
#from model import models
from sklearn.model_selection import KFold
from keras import backend as K

def create_keras_meta(model, batch_size, n_folds, X_train, X_test, y_train):
    
    train_meta = np.zeros(y_train.shape)
    test_meta = np.zeros((X_test.shape[0], 6))
    initial_weights = model.get_weights()
    kf = KFold(n_splits = n_folds, shuffle=True)

    for fold, (train_ix, valid_ix) in enumerate(kf.split(X_train)):
        
        
        x1,x2 = X_train[train_ix], X_train[valid_ix]
        y1,y2 = y_train[train_ix], y_train[valid_ix]
        
        fit_on_val(model, batch_size, x1,x2,y1,y2)
        
        oof_preds = model.predict(x2)
        train_meta[valid_ix] = oof_preds
        
        test_preds = model.predict(X_test)    
        test_meta += 1/n_folds * test_preds
    
        del x1,x2,y1,y2; gc.collect()
        
        #K.clear_session()
        model.set_weights(initial_weights)
        
    return train_meta, test_meta


def fit_on_val(model, batch_size, x1,x2,y1,y2):
    """train until validation loss stops decreasing
    """
           
    best_weights = None
    val_auroc = -np.inf
    best_epochs = -np.inf
    ttl_epochs = 0
    
    epochs_since_improve = 0
    
    while epochs_since_improve < 2:
        model.fit(x1, y1, batch_size, verbose = 1)
        y_preds = model.predict(x2, batch_size)
        
        current_auroc = calc_loss(y2, y_preds)
        ttl_epochs +=1
    
        if current_auroc > np.round(val_auroc, 4):
            
            val_auroc = current_auroc
            best_epochs = ttl_epochs
            best_weights = model.get_weights()
            print('~~val auroc ', current_auroc, best_epochs)
            epochs_since_improve = 0

        else:
            epochs_since_improve += 1
            
        
    
    model.set_weights(best_weights)

    print("returning...", val_auroc, best_epochs)    
    
    return model    
    
def calc_loss(y_true, y_pred):   
    total_auroc = 0
    
    for j in range(6):
        class_auroc = roc_auc_score(y_true[:, j], y_pred[:, j])
        total_auroc += class_auroc

    mean_auroc = (total_auroc + 1e-15) / 6
    
    return mean_auroc