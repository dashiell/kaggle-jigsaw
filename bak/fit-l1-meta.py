import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import log_loss
from skopt import gp_minimize
from sklearn.model_selection import train_test_split
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt.plots import plot_convergence
import dataset
from train_utils import calc_loss


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
y_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train[y_cols].values

#level1_clfs = ['mlstm', 'lstm', 'gru', 'cnn', 'nb-cv' 'lr-tfid']
level1_clfs = ['mlstm', 'lstm', 'gru', 'cnn', 'nb-cv', 'nb-tfid']

X_train_meta = np.zeros((train.shape[0], 6*len(level1_clfs)))
X_test_meta = np.zeros((test.shape[0], 6*len(level1_clfs)))

# add our engineered features to the meta sets
#tr_eng_feats, te_eng_feats = dataset.get_eng_features()

#X_train_meta = np.hstack((X_train_meta, tr_eng_feats))
#X_test_meta = np.hstack((X_test_meta, te_eng_feats))


for i in range(len(level1_clfs)):
    start = i*6
    end = start+6
    
    X_train_meta[:, start:end] = np.load( '../meta/train-{}.npy'.format(level1_clfs[i]) )
    X_test_meta[:, start:end] = np.load( '../meta/test-{}.npy'.format(level1_clfs[i]) )


dim_max_depth = Integer(low = 1, high = 5, name='max_depth')
dim_learning_rate = Real(low = 1e-3, high = 0.3, name='learning_rate')
dim_n_estimators = Integer(low = 50,high = 1500, name='n_estimators')
dim_min_child_weight = Real(low = 0.1, high = 2, name='min_child_weight')
dim_colsample_bytree = Real(low = 0.5, high = 1, name='colsample_bytree')
dim_gamma = Real(low=0, high = 2, name='gamma')
dimensions = [dim_max_depth, 
              dim_learning_rate,
              dim_n_estimators,
              dim_min_child_weight,
              dim_colsample_bytree,
              dim_gamma,
              ]

default_params = [2, 0.15, 549, 1.67, 0.71, 0.05]


def xgb_meta_model(max_depth, learning_rate, n_estimators, min_child_weight, colsample_bytree, gamma):
    
    par = { 'tree_method':'gpu_hist' }
    '''
    model = XGBClassifier(objective='binary:logistic',
                          max_depth=max_depth,
                          learning_rate=learning_rate,
                          n_estimators=n_estimators,
                          min_child_weight=min_child_weight,
                          colsample_bytree=colsample_bytree,
                          gamma=gamma,
                          **par
                          )
    '''
    model = XGBClassifier(objective='binary:logistic', **par)
    model = MultiOutputClassifier(model)
    
    return model

# train to minimize log loss but select models with the best auroc (lowest misclassifications)

@use_named_args(dimensions=dimensions)
def fitness(max_depth, learning_rate, n_estimators, min_child_weight, colsample_bytree, gamma):
    print(max_depth, learning_rate, n_estimators, min_child_weight, colsample_bytree, gamma)
    
    x1,x2,y1,y2 = train_test_split(X_train_meta, y_train, test_size=0.1, shuffle=True)

    model = xgb_meta_model(max_depth, learning_rate, n_estimators, min_child_weight, colsample_bytree, gamma)

    model.fit(x1, y1)
    preds = model.predict_proba(x2)
    preds = np.array(preds)[:,:,1].T

    loss = calc_loss(y2, preds)
    print('  loss', loss)
    
    return loss
 
def fitbest():
    search_results = gp_minimize(func = fitness,
                                 dimensions = dimensions,
                                 #acq_func='EI', # expected improvement
                                 n_calls = 50,
                                 x0=default_params)
    
    plot_convergence(search_results)
    
    #best_params = search_results.x
    #fitness(x=best_params)
    
    # fit the entire dataset on the best 10 fits and use an average of those predictions
    # for our final predictions..
    
    y_preds = np.zeros((test.shape[0], 6))
    
    all_fits = sorted(zip(search_results.func_vals, search_results.x_iters))
    print(all_fits)
'''
    print(all_fits)
    for i in range(10):
        params = all_fits[i][1]
        model = xgb_meta_model(*params)
        model.fit(X_train_meta, y_train)
        preds = np.array(model.predict_proba(X_test_meta))[:,:,1].T
        y_preds += 1/10 * preds

y_preds = fitbest()

'''


#fitbest()


model = xgb_meta_model(*default_params)
model.fit(X_train_meta, y_train)
y_preds = np.array(model.predict_proba(X_test_meta))[:,:,1].T


y_preds_scaled = y_preds ** 1.3 ## 1.4 is best

submit = pd.DataFrame()

submit['id'] = test.loc[:, 'id']

submit = pd.concat([submit, pd.DataFrame(y_preds_scaled, columns=y_cols)], axis=1)
submit.to_csv('submission.csv', index=False)
