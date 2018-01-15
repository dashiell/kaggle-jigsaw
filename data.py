import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
import pickle
#import matplotlib.pyplot as plt

"""
Creates two dictionary with keys:
    
    Data:
        # train and test set (saved to Data.pkl)
        X_train, X_test, y_train
        
        # train, test, and validation (saved to Data-valid.pkl)
        X_valid, y_valid
        
        # column names of the response
        y_cols
        
    ModelParams:
        # paths to best fit model weights
        checkpoint_cnn 
        checkpoint_rnn
        
        # length of padded input vecors, as determined by histogram
        emb_input_seq_len
        
        # size of embedding vector
        emb_out_size
        
        # the shape of the model output (# of columns in y_train)
        model_output_size     
"""

Data = {}
ModelParams = {}

Data['y_cols'] = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

ModelParams['checkpoint_cnn'] = 'checkpoint-cnn.hdf5' 
ModelParams['checkpoint_rnn'] = 'checkpoint-rnn.hdf5'

ModelParams['emb_input_seq_len'] = 120
ModelParams['emb_out_size'] = 256

'''
    Adds X_train, X_test, y_train to the Data dictionary
    Adds emb_vocab_size to the ModelParams dictionary
'''

def create_train_test(nrows_train = None, nrows_test = None):
    
    train = pd.read_csv('../input/train.csv', nrows = nrows_train)
    test = pd.read_csv('../input/test.csv', nrows = nrows_test)
    
    Data['y_train'] = train.loc[:, Data['y_cols']].values
    
    ModelParams['model_output_size'] = Data['y_train'].shape[1]
    
    
    '''
    Generate padded sequences from the train and test comment_text
    '''
    
    # test has 1 nan. 
    test.comment_text.fillna('notextatall', inplace = True)
    
    # all of the raw text data in the training and test set.
    all_comment_text = np.hstack([train.comment_text, test.comment_text])
    
    # converts text to sequences of integers
    tok = text.Tokenizer()
    
    # get a unique integer for each word in the training set.
    tok.fit_on_texts(all_comment_text)
    
    # The number of unique words in raw_text. 
    ModelParams['emb_vocab_size'] = len(tok.word_index) + 1
    
    # integer encode each word in each document (sample). 
    seq_train = tok.texts_to_sequences(train.comment_text)
    seq_test = tok.texts_to_sequences(test.comment_text)
    
    # Use a histogram to determine the max sequence length in order to limit the 
    # network size.  Used to set ModelParams['emb_input_seq_len']
    #doc_lengths = [len(doc) for doc in seq_train] 
    #plt.histogram(doc_lengths, num_bins=1)
    
    # make all sequences the same length for keras by padding them with 0s
    Data['X_train'] = sequence.pad_sequences(seq_train, maxlen = ModelParams['emb_input_seq_len'])
    Data['X_test'] = sequence.pad_sequences(seq_test, maxlen = ModelParams['emb_input_seq_len'])
    
    pickle.dump(ModelParams, open('ModelParams.pkl', 'wb'))
    pickle.dump(Data, open('Data.pkl', 'wb'))

'''
    Adds X_valid, y_valid to the Data dictionary.
'''

def create_valid(test_size = 0.05, random_state = True):
    Data = pickle.load(open('Data.pkl', 'rb'))
    
    any_positive_category = np.sum(Data['y_train'], axis = 1)

    Data['X_train'], Data['X_valid'], Data['y_train'], Data['y_valid'] = train_test_split(
        Data['X_train'], 
        Data['y_train'],
        test_size = test_size,
        stratify = any_positive_category,
        random_state = random_state
        )
    pickle.dump(Data, open('Data-valid.pkl', 'wb'))

'''
    returns the Modelparams and Data dictionaries
'''

def load(with_validation_set = True):
    ModelParams = pickle.load(open('ModelParams.pkl', 'rb'))
    
    data_path = 'Data.pkl'
    if with_validation_set:
        data_path = 'Data-valid.pkl'
    
    Data = pickle.load(open(data_path, 'rb'))
    
    return ModelParams, Data
    
    
ModelParams, Data = load()