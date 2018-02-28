'''
Find the best hyperparameters for our GRU model using scikit's bayesian optimization
'''

import dataset
from model import models
from skopt.utils import use_named_args
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.plots import plot_convergence
import pandas as pd

import gc
from keras import backend as K

def get_dimensions():
    
    dim_batch_size = Integer(low = 64, high = 512, name='batch_size') 
    dim_n_dense_layers = Integer(low = 1, high = 3, name = 'n_dense_layers')
    dim_n_dense_outputs = Integer(low = 16, high = 128, name = 'n_dense_outputs')
    dim_dropout_rate = Real(low = 0.01, high = 0.7, name = 'dropout_rate')
    dim_l2_rate = Real(low = 1e-15, high = 1e-2, prior = 'log-uniform', name='l2_rate')
    
    dimensions = [
                  dim_batch_size,
                  dim_n_dense_layers,
                  dim_n_dense_outputs,
                  dim_dropout_rate,
                  dim_l2_rate,
                  ]

    return dimensions


@use_named_args(dimensions = get_dimensions())
def fitness(batch_size, n_dense_layers, n_dense_outputs, 
                dropout_rate, l2_rate):
       
    x1,x2,y1,y2 = dataset.get_train_valid(X_train, y_train, test_size=0.1)
    
    model = models.lstm_model(EMB_MATRIX, SEQUENCE_LEN)
    #m.build_meta(n_dense_layers, n_dense_outputs, dropout_rate, l2_rate)
    model = models.fit_on_val(model, 128, x1,x2,y1,y2)
    y_pred = model.predict(x2)
    validation_loss = models.calc_loss(y2, y_pred)
    print('val loss', validation_loss)
    del model; gc.collect()
    K.clear_session()
    
    return validation_loss

X_train, X_test, y_train = dataset.get_keras_data()

EMB_MATRIX = dataset.get_emb_matrix2()
SEQUENCE_LEN = dataset.SEQUENCE_LEN




#default_params = [128, 2, 100, 0.4, 5.80236e-05]
#fitness_gru(default_params)

search_results = gp_minimize(func = fitness,
                             dimensions = get_dimensions(),
                             acq_func='EI', # expected improvement
                             n_calls = 11,
                             #x0=default_params
                             )
                            
plot_convergence(search_results)
'''
best_params = search_results.x

print("best_params: ", best_params)

# get the best 5 fits. and make an ensembled meta dataset from them.
all_fits = sorted(zip(search_results.func_vals, search_results.x_iters))

fitdata = {'loss' : [i[0] for i in all_fits],
        'batch_size' : [i[1][0] for i in all_fits],
        'params' : [i[1][1:] for i in all_fits]
}

pd.DataFrame(fitdata).to_csv('meta-model-results-new.csv', index=False)

'''

