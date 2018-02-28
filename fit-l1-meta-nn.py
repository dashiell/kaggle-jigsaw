import numpy as np
import dataset
import train_utils
from sklearn.model_selection import KFold
from model.models import meta_model
from keras import backend as K
import gc
import pandas as pd
#level1_clfs = ['mlstm', 'lstm', 'gru', 'cnn', 'nb-cv' 'lr-tfid']
level1_clfs = ['mlstm', 'lstm', 'gru', 'cnn', 'nb-cv', 'nb-tfid', 'gru-9845']

X_train_meta = np.zeros((dataset.nrows_train, 6*len(level1_clfs)))
X_test_meta = np.zeros((dataset.nrows_test, 6*len(level1_clfs)))


# add our engineered features to the meta sets
tr_oh, te_oh = dataset.get_oh_features()
tr_c, te_c = dataset.get_cont_features()

X_train_meta = np.hstack((X_train_meta, tr_oh, tr_c))
X_test_meta = np.hstack((X_test_meta, te_oh, te_c))

for i in range(len(level1_clfs)):
    start = i*6
    end = start+6
    
    X_train_meta[:, start:end] = np.load( '../meta/train/{}.npy'.format(level1_clfs[i]) )
    X_test_meta[:, start:end] = np.load( '../meta/test/{}.npy'.format(level1_clfs[i]) )




model = meta_model(X_train_meta.shape[1])
train_meta, test_meta = train_utils.create_keras_meta(model, 128, 10, X_train_meta, X_test_meta, dataset.y_train)

np.save('../meta-l2/train/nn.npy', train_meta)
np.save('../meta-l2/test/nn.npy', test_meta)
    

    

'''
y_preds = np.zeros( shape=(dataset.nrows_test,6) )

n_folds = 10

kf = KFold(n_splits = n_folds, shuffle=True)
for fold, (train_ix, val_ix) in enumerate(kf.split(X_train_meta)):
    print("\t\t\t\t\t\t\t\tfold", fold)        
    model = meta_model(X_train_meta.shape[1])
    
    x1, x2 = get_keras_data(train_ix, val_ix)  
    
    train_utils.fit_on_val(model, 
                           batch_size=128, 
                           x1=x1, 
                           x2=x2,
                           y1=dataset.y_train[train_ix],
                           y2=dataset.y_train[val_ix]


                           )
    y_preds += 1/n_folds * model.predict(X_test_meta, batch_size=128)
    
    del model; gc.collect()
    K.clear_session()
'''
'''
model = meta_model()
#x1,x2,y1,y2 = train_test_split(X_train_meta, dataset.y_train, test_size=0.1)
#train_utils.fit_on_val(model, 128, x1,x2,y1,y2)
model.fit(X_train_meta, dataset.y_train, batch_size=128, epochs=4)
y_preds = model.predict(X_test_meta, batch_size=128)

'''
submit = pd.DataFrame()

submit['id'] = dataset.test.loc[:, 'id']

submit = pd.concat([submit, pd.DataFrame(y_preds, columns=dataset.y_cols)], axis=1)
submit.to_csv('submission.csv', index=False)