import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from sklearn.preprocessing import OneHotEncoder, scale
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk_utils

import pickle
import re



EMBED_DICT_PATH = '../input/embed_dict.pkl'
SEQUENCE_LEN = 150
CHAR_SEQ_LEN = 512
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
test.comment_text.fillna('notextatall', inplace = True)

y_cols =  ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

'''
def get_keras_dict(comment_text):
    """data for keras"""
    
    embed_dict = get_embed_dict()
    
    # integer encode each word in each document / sample. 
    seq_comment_text = embed_dict['tokenizer'].texts_to_sequences(comment_text)
    
    # make all sequences the same length for keras by padding them with 0s
    padded_seq = sequence.pad_sequences(seq_comment_text, maxlen = embed_dict['emb_input_seq_len'])
    
    X = {
            'comment_text' : padded_seq
       
            }
    return X
'''
def get_train_valid(X_train, y_train, test_size = 0.1):
    
    X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, 
            y_train,
            test_size = test_size,
            shuffle = True
        )
    
    return X_train, X_valid, y_train, y_valid

def get_eng_features():
    """feature enineering"""
    
    df = pd.concat([train, test], axis=0)
    
    num_words = [len( re.findall("\S+", t )) for t in df.comment_text]
    num_words = scale(num_words).reshape(-1,1)
    
    pct_caps = df.comment_text.str.count(r'[A-Z]').values / df.comment_text.str.len().values
    pct_caps = scale(pct_caps).reshape(-1,1)
    
    # One Hot Features
    enc = OneHotEncoder(sparse=False)
    
    has_ipaddr = df.comment_text.str.contains(r'\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}').values.reshape(-1,1)
    has_ipaddr = enc.fit_transform(has_ipaddr)
    
    
    has_utc = df.comment_text.str.contains(r'(UTC)').values.reshape(-1,1)
    has_utc = enc.fit_transform(has_utc)

    has_url = df.comment_text.str.contains(r'http[s]?://').values.reshape(-1,1)
    has_url = enc.fit_transform(has_url)
    
    has_quote = df.comment_text.str.contains(r'\"').values.reshape(-1,1)
    has_quote = enc.fit_transform(has_quote)
    
    feats = np.hstack([num_words, pct_caps]) ### TODO: add other features
    
    nrows_train = train.shape[0]
    tr_feats = feats[:nrows_train]
    te_feats = feats[nrows_train:]
    
    return tr_feats, te_feats

def _create_embedding_matrix(fpath, tokenizer, dim):
    
    # The number of unique words in raw_text. 
    vocab_size = len(tokenizer.word_index) + 1
    
    embeddings_index = {}
    f = open(fpath)
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    # prepare embedding matrix
    embedding_matrix = np.zeros((vocab_size, dim))
    
    words_missing = 0
    
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            words_missing += 1
            
    print('missing from {}: {} words'.format(fpath, words_missing))    
    
    return embedding_matrix


def _create_embed_dict():
    """"creates the embedding dictionary with pretrained glove embedding matrix weights"""
    
    all_comment_text = np.hstack([train.comment_text.values, test.comment_text.values])
    
    # converts text to sequences of integers
    tokenizer = text.Tokenizer()
    
    # get a unique integer for each word in the training set.
    tokenizer.fit_on_texts(all_comment_text)
    
    emb_matrix1 = _create_embedding_matrix('../input/glove.840B.300d.txt', tokenizer, 300)
    emb_matrix2 = _create_embedding_matrix('../input/glove.twitter.27B.200d.txt', tokenizer, 200)

    # Use a histogram to determine the max sequence length in order to limit the 
    # network size.  Used to set embed_dict['emb_input_seq_len']
    #doc_lengths = [len(doc) for doc in seq_train] 
    #plt.histogram(doc_lengths, num_bins=1)
            
    embed_dict = {'tokenizer' : tokenizer,
                  'embedding_matrix1' : emb_matrix1,
                  'embedding_matrix2' : emb_matrix2,
                  }
        
    pickle.dump(embed_dict, open(EMBED_DICT_PATH, 'wb'))

    return embed_dict

def get_tfidvecs(max_features=None):
    
    tfv = TfidfVectorizer(min_df=3,  max_features=max_features, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
            stop_words = 'english')
    
    tfv.fit(train.comment_text, test.comment_text)
    
    X_train = tfv.transform(train.comment_text)
    X_test = tfv.transform(test.comment_text)
        
    return X_train, X_test, train[y_cols].values

def get_countvecs():
    cv = CountVectorizer(analyzer='word',token_pattern=r'\w{1,}',
                         ngram_range=(1, 3), stop_words = 'english')

    cv.fit(train.comment_text, test.comment_text)
    
    X_train = cv.transform(train.comment_text)
    X_test = cv.transform(test.comment_text)
        
    return X_train, X_test, train[y_cols].values
    

def get_keras_data():
    tokenizer = get_embed_dict('tokenizer')
    
    # integer encode each word in each document / sample. 
    X_train = tokenizer.texts_to_sequences(train.comment_text.values)
    X_test = tokenizer.texts_to_sequences(test.comment_text.values)
    
    # make all sequences the same length for keras by padding them with 0s
    X_train = sequence.pad_sequences(X_train, maxlen = SEQUENCE_LEN)
    X_test = sequence.pad_sequences(X_test, maxlen = SEQUENCE_LEN)

    return X_train, X_test, train[y_cols].values

def get_char_data():
    X_train = np.zeros((train.shape[0], CHAR_SEQ_LEN))
    X_test = np.zeros((test.shape[0], CHAR_SEQ_LEN))
    # .str.lower()
    for i, comment in enumerate(train.comment_text.values):
        X_train[i] = nltk_utils.get_comment_ids(comment, CHAR_SEQ_LEN)
        
    for i, comment in enumerate(test.comment_text.values):
        X_test[i] = nltk_utils.get_comment_ids(comment, CHAR_SEQ_LEN)

    return X_train, X_test, train[y_cols].values
    

def get_train_test():
    return train['comment_text'], test['comment_text'], train[y_cols].values
    

def get_embed_dict(key):
    """contains the embedding_matrix and fitted tokenizer object"""
    
    try: 
        embed_dict = pickle.load(open(EMBED_DICT_PATH, 'rb'))
    except FileNotFoundError:
        embed_dict = _create_embed_dict()
    
    return embed_dict[key]

def get_tokenizer():
    return get_embed_dict('tokenizer')
    

def get_emb_matrix1():
    return get_embed_dict('embedding_matrix1')

def get_emb_matrix2():
    return get_embed_dict('embedding_matrix2')

