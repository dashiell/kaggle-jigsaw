'''
toxic            15294
severe_toxic      1595
obscene           8449
threat             478
insult            7877
identity_hate     1405
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import log_loss

train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')

def calc_loss(y_true, y_pred):
    
    total_loss = 0
    
    for j in range(6):
        class_logloss = log_loss(y_true[:, j], y_pred[:, j])
        total_loss += class_logloss

    total_loss /= 6
    
    return total_loss

y_train = train.iloc[:,2:8]

tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 4), use_idf=1, smooth_idf=1, sublinear_tf=1,
            stop_words = 'english')

x1,x2,y1,y2 = train_test_split(train.comment_text, y_train, test_size=0.1)

tfv.fit(train.comment_text)

x1 = tfv.transform(x1)
x2 = tfv.transform(x2)

model = XGBClassifier(max_depth=7, n_estimators=500, colsample_bytree=0.8, 
                        subsample=0.8, nthread=10, learning_rate=0.1)
model = MultiOutputClassifier(model)
model.fit(x1, y1)
y_pred = model.predict_proba(x2)

import numpy as np
y_new = np.array(y_pred)[:,:,1].T
calc_loss(y2, y_new)

log_loss[y_pred][0]