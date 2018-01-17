import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
import pickle
from os import path
#import matplotlib.pyplot as plt

##### TODO
# add features:
# percentage_caps - percentage of the sentence that is in caps
# contains_url 
# contains_country 
# contains_ethnicity (slavic, white)
# contains_badword
# has negative word (... or do sentiment analysis...)
# length of original text
# uniqueness: number of unique words in text / total number of words
# of words not in embedding / total words
# contains_ip_addr 
# contains_condescending words
# contains_date ( like (UTC) see examples... few negative)
# contanis_nice ( e.g. Hello, Welcome, Hey, )

GLOVE_PATH = '../input/glove.840B.300d.txt'
TRAIN_PATH = '../input/train.csv'
TEST_PATH = '../input/test.csv'
MODEL_PARAMS_PATH = '../input/model_params.pkl'
DATA_TT_PATH = '../input/data_novalidationset.pkl' # train, test
DATA_TTV_PATH = '../input/data_validationset.pkl' # train, test, validation

LOGGING = True

'''
 To get the data and model_params dictionaries, call:
     data, model_params = load(validation_size, use_glove)
'''


data = {
        # padded sequence vectors; nparray with shape (nrows_train, model_params['emb_input_seq_len'])
        'X_train' : None, 
        'X_test' : None,
        'X_valid' : None, 
        # nparray of shape (nrows_train, len(y_cols))
        'y_train' : None, 
        'y_valid' : None, 
        
        'y_cols' : ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], # column names of the response
        
        'nrows_train' : 0,
        'nrows_test' : 0,
        'nrows_valid' : 0,
        
        'embedding_matrix' : None   # pre-trained glove word embedding matrix
        }
        
model_params = {
        'emb_input_seq_len' : 150, # length of padded input vecors, as determined by histogram
        'emb_out_size' : 300, # size of embedding vector
        'emb_vocab_size' : None,
        'use_glove' : True # use pre-trained word embeddings
        }


def _create(use_glove):
    '''Adds X_train, X_test, y_train to the data dictionary.
    Adds emb_vocab_size to the model_params dictionary.
    
    Parameters
    ----------
    * `nrows` [`int`]:
        Number of rows to read from the training & test set
    * `use_glove` [`bool`]:
        Use glove embeddings if true
    '''
    if LOGGING:
        print('creating train and test set, word emb')
    
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    
    data['y_train'] = train.loc[:, data['y_cols']].values
    
    data['nrows_train'] = train.shape[0]
    data['nrows_test'] = test.shape[0]
    
    #### 
    #Generate padded sequences from the train and test comment_text
    ####
    
    # test has 1 nan. 
    test.comment_text.fillna('notextatall', inplace = True)
    
    # all of the raw text data in the training and test set.
    all_comment_text = np.hstack([train.comment_text, test.comment_text])
    
    # converts text to sequences of integers
    tok = text.Tokenizer()
    
    # get a unique integer for each word in the training set.
    tok.fit_on_texts(all_comment_text)
    
    # The number of unique words in raw_text. 
    model_params['emb_vocab_size'] = len(tok.word_index) + 1
    
    # integer encode each word in each document (sample). 
    seq_train = tok.texts_to_sequences(train.comment_text)
    seq_test = tok.texts_to_sequences(test.comment_text)
    
    # Use a histogram to determine the max sequence length in order to limit the 
    # network size.  Used to set model_params['emb_input_seq_len']
    #doc_lengths = [len(doc) for doc in seq_train] 
    #plt.histogram(doc_lengths, num_bins=1)
    
    # make all sequences the same length for keras by padding them with 0s
    data['X_train'] = sequence.pad_sequences(seq_train, maxlen = model_params['emb_input_seq_len'])
    data['X_test'] = sequence.pad_sequences(seq_test, maxlen = model_params['emb_input_seq_len'])
    
    get_embedding_matrix(tok)
       
    pickle.dump(model_params, open(MODEL_PARAMS_PATH, 'wb'))
    pickle.dump(data, open(DATA_TT_PATH, 'wb'))


def _create_validation_set(test_size):
    '''
    Adds X_valid, y_valid to the data dictionary.
    '''
    if LOGGING:
        print('creating validation set')
    
    data = pickle.load(open(DATA_TT_PATH, 'rb'))
    
    any_positive_category = np.sum(data['y_train'], axis = 1)

    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = train_test_split(
        data['X_train'], 
        data['y_train'],
        test_size = test_size,
        stratify = any_positive_category,
        )
    
    data['nrows_train'] = data['X_train'].shape[0]
    data['nrows_valid'] = data['X_valid'].shape[0]
    
    pickle.dump(data, open(DATA_TTV_PATH, 'wb'))

'''
    returns the model_params and data dictionaries
'''

def get_embedding_matrix(tokenizer):
    """"get the pretrained glove embedding matrix weights
    
    params
    ------
    * `tokenizer` [`Tokenizer`]
    
    returns
    -------
    embedding matrix of glove weights
    """
    
    #if path.exists(GLOVE_WEIGHTS):
    #    print('getting glvoe weights from tokenized words')
    #    data['embedding_matrix'] = np.load(GLOVE_WEIGHTS)
        
    embeddings_index = {}
    f = open('../input/glove.840B.300d.txt')
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    # prepare embedding matrix
    embedding_matrix = np.zeros((model_params['emb_vocab_size'], 
                                model_params['emb_out_size']))
    
    words_missing = 0
    
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            words_missing += 1
            
    print('missing from glove: {} words'.format(words_missing))
    
    #np.save(GLOVE_WEIGHTS, embedding_matrix)
    
    data['embedding_matrix'] = embedding_matrix



def load(validation_size = .05, use_glove = True, force_rebuild = False):
    """loads the train, test, and optional validation sets.  Also loads the 
    glove word embedding matrix, if use_glove is True
    
    ** use force_rebuild=True if you changed something like emb_input_seq_len or 
    added new features
    """
    data_path = DATA_TTV_PATH if validation_size > 0 else DATA_TT_PATH    

    if not path.exists(data_path) or force_rebuild:
        # create the data dictionary with training set, test set, embeddings
        _create(use_glove)
    
    data = pickle.load(open(data_path, 'rb'))
    
    if validation_size > 0:
        _create_validation_set(validation_size)
        
    model_params = pickle.load(open(MODEL_PARAMS_PATH, 'rb'))
    
    if use_glove:
        model_params['use_glove'] = True
    
    return data, model_params