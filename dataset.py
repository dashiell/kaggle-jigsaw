import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
from os import path
import gc
import re
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
EMBED_DICT_PATH = '../input/embed_dict.pkl'


y_cols =  ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
       
embed_dict = {
        'emb_input_seq_len' : 150, # length of padded input vecors, as determined by histogram
        'emb_out_size' : 300, # size of embedding vector
        'emb_vocab_size' : None,
        'use_glove' : True, # use pre-trained word embeddings
        'embedding_matrix' : None,
        'tokenizer': None,
        }


def get_keras_dict(embed_dict, comment_text):
    """data for keras"""
    
    # integer encode each word in each document / sample. 
    seq_comment_text = embed_dict['tokenizer'].texts_to_sequences(comment_text)
    
    # make all sequences the same length for keras by padding them with 0s
    padded_seq = sequence.pad_sequences(seq_comment_text, maxlen = embed_dict['emb_input_seq_len'])
    
    #encoder = OneHotEncoder()
    
    has_utc = [bool(re.search(r'(UTC)', t)) for t in comment_text]
    #has_utc = comment_text.str.contains('(UTC)').values.reshape(-1,1)
    #has_utc = encoder.fit_transform(has_utc).toarray()
    
    #### TODO: normalize
    #pct_caps = comment_text.str.count(r'[A-Z]') / comment_text.str.len()
    
    #has_ipaddr = comment_text.str.contains(r'\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}')
    
    #### TODO: add list of ethnicities..
    #has_ethnicity = comment_text.str.contains(r'jew', case=False)
    X = {
            'comment_text' : padded_seq,
            'has_utc' : np.array(has_utc),
            #'pct_caps' : pct_caps,
            #'has_ethnicity' : has_ethnicity,
            #'has_ipaddr' : has_ipaddr
            }
    return X

def get_train_valid(X_train, y_train, test_size):
    
    any_positive_category = np.sum(y_train, axis = 1)

    X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, 
            y_train,
            test_size = test_size,
            stratify = any_positive_category,
        )
    
    return X_train, X_valid, y_train, y_valid


'''
    returns the embed_dict and data dictionaries
'''

def _create_embed_dict(all_comment_text):
    """"get the pretrained glove embedding matrix weights"""
    

    # converts text to sequences of integers
    tokenizer = text.Tokenizer()
    
    # get a unique integer for each word in the training set.
    tokenizer.fit_on_texts(all_comment_text)
    
    # The number of unique words in raw_text. 
    embed_dict['emb_vocab_size'] = len(tokenizer.word_index) + 1
    
    # Use a histogram to determine the max sequence length in order to limit the 
    # network size.  Used to set embed_dict['emb_input_seq_len']
    #doc_lengths = [len(doc) for doc in seq_train] 
    #plt.histogram(doc_lengths, num_bins=1)
            
    embeddings_index = {}
    f = open('../input/glove.840B.300d.txt')
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    # prepare embedding matrix
    embedding_matrix = np.zeros((embed_dict['emb_vocab_size'], 
                                embed_dict['emb_out_size']))
    
    words_missing = 0
    
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            words_missing += 1
            
    print('missing from glove: {} words'.format(words_missing))
        
    embed_dict['tokenizer'] = tokenizer
    embed_dict['embedding_matrix'] = embedding_matrix
    
    pickle.dump(embed_dict, open(EMBED_DICT_PATH, 'wb'))



def load(use_glove = True):
    """loads the train, test.  Also loads the 
    glove word embedding matrix, if use_glove is True
    
    """ 
    train = pd.read_csv(TRAIN_PATH)
    X_train = train.comment_text.values
    y_train = train.loc[:, y_cols].values
    test = pd.read_csv(TEST_PATH)
    test.comment_text.fillna('notextatall', inplace = True)
    X_test = test.comment_text.values
    
    del train; del test; gc.collect()
    
    # create the dictionary if it's not already there
    if not path.exists(EMBED_DICT_PATH):
        all_comment_text = np.hstack((X_train, X_test))
        _create_embed_dict(all_comment_text)
    
    embed_dict = pickle.load(open(EMBED_DICT_PATH, 'rb'))
        
    embed_dict['use_glove'] = use_glove
        
    #X_train, X_valid, y_train, y_valid = get_train_valid(X_train, y_train, 0.1)
    
    #data['X_train'] = _get_keras_dict(comment_text_train, tok)
    #data['X_valid'] = _get_keras_dict(comment_text_valid, tok)
    #data['X_test'] = _get_keras_dict(test.comment_text, tok)
    #data['y_train'] = y_train
    #data['y_valid'] = y_valid
              
    
    return X_train, X_test, y_train, embed_dict