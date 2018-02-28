"""Create L1 meta-train, meta-test """

import dataset
from model import models
from keras.utils import plot_model

import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import gc
import train_utils


"""Create meta models for keras word based models with pre-trained embeddings"""

def create_meta_keras_word():

    
    embedding_mat = dataset.get_emb_matrix1()
    seq_len = dataset.SEQUENCE_LEN
    
    #X_train, X_test, y_train = dataset.get_keras_data()
    
    model = models.lstm_model(embedding_mat, seq_len)
    train_meta, test_meta = train_utils.create_keras_meta(model, 128, 10, X_train, X_test, y_train)
        
    #tr_path = '../meta/train-{}.npy'.format(mdict['func'].__name__)
    #te_path = '../meta/test-{}.npy'.format(mdict['func'].__name__)
            
    #np.save(tr_path, train_meta)
    #np.save(te_path, test_meta)




X_train, X_test, y_train = dataset.get_char_data()

model = models.vdcnn_model()
x1,x2,y1,y2 = dataset.get_train_valid(X_train, y_train)
plot_model(model, show_shapes=True)
train_utils.fit_on_val(model, 128, x1,x2,y1,y2)
del model; gc.collect()

'''
def create_simple_meta(X_train, X_test, y_train, n_folds, model, name):
    """creating naive bayes meta datasets"""
        
    train_meta = np.zeros(y_train.shape)
    test_meta = np.zeros((X_test.shape[0], 6))    
    
    kf = KFold(n_splits = n_folds, shuffle=True)


    for fold, (train_ix, val_ix) in enumerate(kf.split(X_train)):
        
        model.fit(X_train[train_ix], y_train[train_ix])
    
        oof_preds = np.array(model.predict_proba(X_train[val_ix]))[:,:,1].T
        train_meta[val_ix] = oof_preds
        test_preds = model.predict(X_test)
        print(models.calc_loss(y_train[val_ix], oof_preds))
        
        test_meta += 1/n_folds * test_preds
    
    np.save('../meta/train-{}.npy'.format(name), train_meta)
    np.save('../meta/test-{}.npy'.format(name), test_meta)


#create_keras_meta()

#create_simple_meta(*dataset.get_countvecs(), 10, MultiOutputClassifier(MultinomialNB()), 'nb-cv')
#create_simple_meta(*dataset.get_tfidvecs(), 10, MultiOutputClassifier(MultinomialNB()), 'nb-tfid')
#create_simple_meta(*dataset.get_tfidvecs(max_features=3500), 10, MultiOutputClassifier(LogisticRegression(C=1.0)), 'lr-tfid')

'''

'''
import pandas as pd
test = pd.read_csv('../input/test.csv')

y_preds = te_meta_gru# (y_pred_cnn + y_pred_lstm + y_pred_mlstm) / 3

'''
'''
model = models.vdcnn_model()
X_train,X_test, y_train = dataset.get_char_data()
plot_model(model, show_shapes=True)
model.fit(X_train, y_train, 128, epochs=2)
y_preds = model.predict(X_test)
del model; gc.collect()

import pandas as pd
test = pd.read_csv('../input/test.csv')
submit = pd.DataFrame()
    
submit['id'] = test.loc[:, 'id']

submit = pd.concat([submit, pd.DataFrame(y_preds, columns=dataset.y_cols)], axis=1)
submit.to_csv('submission.csv', index=False)
'''