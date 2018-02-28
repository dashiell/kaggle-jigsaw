import numpy as np
from sklearn.model_selection import StratifiedKFold
import Models

n_splits = 10

X = np.arange(0,160).reshape(40,4)
y = np.repeat([0,1,2,3], 10)

skf = StratifiedKFold(n_splits)

#for i, (tr_ix, te_ix) in enumerate(skf.split(X, y)):
#    print(i)



