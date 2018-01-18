import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
from os import path
import gc
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
DATA_PATH = '../input/data.pkl' # train, test, validation


LOGGING = True

'''
 To get the data and model_params dictionaries, call:
     data, model_params = load(validation_size, use_glove)
'''


data = {
        # padded sequence vectors; nparray with shape (nrows_train, model_params['emb_input_seq_len'])
        'X_train' : {}, 
        'X_test' : {}, 
        'X_valid' : {}, 
        # nparray of shape (nrows_train, len(y_cols))
        'y_train' : None, 
        'y_valid' : None, 
        
        # column names of the response
        'y_cols' : ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'],
              
        'embedding_matrix' : None   # pre-trained glove word embedding matrix
        }
        
model_params = {
        'emb_input_seq_len' : 150, # length of padded input vecors, as determined by histogram
        'emb_out_size' : 300, # size of embedding vector
        'emb_vocab_size' : None,
        'use_glove' : True # use pre-trained word embeddings
        }

def _create_features(df):
    
    # 2.8% of train text with (UTC) has positive y. 10.2% with no UTC has positive.
    has_utc = pd.get_dummies( df['comment_text'].str.contains('(UTC)') )
        
    pd.concat([df, has_utc])

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
    y_train = train.loc[:, data['y_cols']].values
    test = pd.read_csv(TEST_PATH)
    test.comment_text.fillna('notextatall', inplace = True)
    

    #### 
    #Generate padded sequences from the train and test comment_text
    ####
        
    # all of the raw text data in the training and test set.
    all_comment_text = pd.concat((train.comment_text, test.comment_text), axis=0)
        
    # converts text to sequences of integers
    tok = text.Tokenizer()
    
    # get a unique integer for each word in the training set.
    tok.fit_on_texts(all_comment_text)
    
    # The number of unique words in raw_text. 
    model_params['emb_vocab_size'] = len(tok.word_index) + 1
    

    # Use a histogram to determine the max sequence length in order to limit the 
    # network size.  Used to set model_params['emb_input_seq_len']
    #doc_lengths = [len(doc) for doc in seq_train] 
    #plt.histogram(doc_lengths, num_bins=1)

    
    comment_text_train, comment_text_valid, y_train, y_valid = _get_validation_set(
            train.comment_text, y_train, 0.1
            )
    
    data['X_train'] = _get_keras_dict(comment_text_train, tok)
    data['X_valid'] = _get_keras_dict(comment_text_valid, tok)
    data['X_test'] = _get_keras_dict(test.comment_text, tok)
    data['y_train'] = y_train
    data['y_valid'] = y_valid
        
    if use_glove:
        get_embedding_matrix(tok)
       
    pickle.dump(model_params, open(MODEL_PARAMS_PATH, 'wb'))
    pickle.dump(data, open(DATA_PATH, 'wb'))


def _get_keras_dict(comment_text, tokenizer):
    # integer encode each word in each document / sample. 
    seq_comment_text = tokenizer.texts_to_sequences(comment_text)
    
    # make all sequences the same length for keras by padding them with 0s
    padded_seq = sequence.pad_sequences(seq_comment_text, maxlen = model_params['emb_input_seq_len'])
    
    #encoder = OneHotEncoder()
    has_utc = comment_text.str.contains('(UTC)').values.reshape(-1,1)
    #has_utc = encoder.fit_transform(has_utc).toarray()
    
    #### TODO: normalize
    pct_caps = comment_text.str.count(r'[A-Z]') / comment_text.str.len()
    
    has_ipaddr = comment_text.str.contains(r'\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}')
    
    #### TODO: add list of ethnicities..
    has_ethnicity = comment_text.str.contains(r'jew', case=False)
    X = {
            'comment_text' : padded_seq,
            'has_utc' : has_utc,
            'pct_caps' : pct_caps,
            'has_ethnicity' : has_ethnicity,
            'has_ipaddr' : has_ipaddr
            }
    return X

def _get_validation_set(X_train, y_train, test_size):
    
    any_positive_category = np.sum(y_train, axis = 1)

    X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, 
            y_train,
            test_size = test_size,
            stratify = any_positive_category,
        )
    
    return X_train, X_valid, y_train, y_valid


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

    if not path.exists(DATA_PATH) or force_rebuild:
        # create the data dictionary with training set, test set, embeddings
        _create(use_glove)
    
    data = pickle.load(open(DATA_PATH, 'rb'))
    model_params = pickle.load(open(MODEL_PARAMS_PATH, 'rb'))
    
    if use_glove:
        model_params['use_glove'] = True
    
    print('finished loading')
    
    return data, model_params