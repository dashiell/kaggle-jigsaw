"""
Fit the model on the training set, test on validation set
"""
import dataset
import models
from keras.callbacks import EarlyStopping, ModelCheckpoint

CNN_CHECKPOINT = 'cnn_checkpoint.hdf5'
RNN_CHECKPOINT = 'rnn_checkpoint.hdf5'

def fit(model, checkpoint_path):

    # define callbacks
    cb_earlystop = EarlyStopping(patience=3)
    cb_checkpoint = ModelCheckpoint(checkpoint_path, 
                                 monitor='val_loss', 
                                 verbose=1,
                                 save_best_only=True, 
                                 mode='min')
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])   

    model.fit(data['X_train'], data['y_train'], 
              epochs = 20, 
              batch_size = 128,  
              callbacks = [cb_earlystop, cb_checkpoint],
              validation_data = (data['X_valid'], data['y_valid']),
              shuffle = True
            )
    
    
    '''

    '''
    
# load the model_params and data dictionaries (see data.py)
data, model_params = dataset.load(validation_size = 0.1, use_glove = True)

# build the model
cnn_model = models.build_cnn_model(model_params, data)
rnn_model = models.build_rnn_model(model_params, data)

# fit the model

fit(cnn_model, CNN_CHECKPOINT)
fit(rnn_model, RNN_CHECKPOINT)


####
import pandas as pd
test = pd.read_csv('../input/test.csv')

y_preds = cnn_model.predict(data['X_test'], batch_size=128)
    
submit = pd.DataFrame()
    
submit['id'] = test.loc[:, 'id']
    
submit = pd.concat([submit, pd.DataFrame(y_preds, columns=data['y_cols'])], axis=1)
submit.to_csv('submission.csv', index=False)