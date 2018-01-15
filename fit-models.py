"""
Fit the model on the training set, test on validation set
"""
import data
import models

#data.create_train_test()
#data.create_train_test(nrows_train=20000, nrows_test=50)
#data.create_valid(test_size=0.2)


def fit(model):

    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])   
    
    model.fit(Data['X_train'], Data['y_train'], 
              epochs = 1, 
              batch_size = 128,  
              #callbacks = [checkpoint_cb],
              validation_data = (Data['X_valid'], Data['y_valid']),
              shuffle = True
            )
    '''
    y_preds = model.predict(padded_seq_test, batch_size=128)
    
    submit = pd.DataFrame()
    
    submit['id'] = test.loc[:, 'id']
    
    submit = pd.concat([submit, pd.DataFrame(y_preds, columns=y_cols)], axis=1)
    submit.to_csv('submission.csv', index=False)
    '''



# load the ModelParams and Data dictionaries (see data.py)
ModelParams, Data = data.load(with_validation_set = True)

# specify the model to build
model = models.build_rnn_model(ModelParams)

# fit the model
fit(model)


