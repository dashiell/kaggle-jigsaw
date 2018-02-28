from tqdm import tqdm
import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from sklearn.preprocessing import OneHotEncoder, normalize, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import pickle
import re, codecs

import nltk_utils

EMBED_DICT_PATH = '../input/embed_dict.pkl'
WORD_SEQ_LEN = 150
CHAR_SEQ_LEN = 512

def preproc():
    train = pd.read_csv('../input/train.csv')
    test = pd.read_csv('../input/test.csv')
    
    repl = {
    "i'm" : "i am",
    "can't" : "can not",
    "who've" : "who have",
    "wasn't" : "was not",
    "you'd" : "you will",
    "don't" : "do not",
    "i'll" : "i will",
    "i've" : "i have",
    "i'd" : "i will",
    "isn't" : "is not",
    "it's" : "it is",
    "doesn't" : "does not",
    "there's" : "there is",
    "you've" : "you have",
    "he's" : "he is",
    "wasn't" : "was not",
    "won't" : "will not",
    "we're" : "we are",
    "cock" : "penis",
    "npov" : "understanding",
    "youfuck" : "you fuck",
    "haven't" : "have not",
    "wouldn't" : "would not",
    "aren't" : "are not",
    "they're" : "they are",
    "shouldn't" : "should not",
    "fucksex" : "fuck sex",
    "niggors" : "niggers",
    "b00ll00x" : "bullshit",
    "bonergasm" : "sex",
    "donkeysex" : "sex",
    "mothjer" : "mother",
    "phck" : "fuck",
    "world's" : "world is",
    "fggt" : "faggot",
    "faggt" : "faggot",
    "fdffe7" : "faggot",
    "vandalizer" : "asshole",
    "faggt" : "faggot",
    "titoxd" : "sex",
    "niggerjew" : "nigger",
    "gayyour" : "you are gay",
    "cuntnlu" : "cunt",
    "we'd" : "we would",
    "offfuck" : "fuck off",
    "vaginapenis" : "dick",
    "'fuck" : "fuck",
    "deneid" : "denied",
    "cuntliz" : "cunt",
    "motherfuckerdie" : "die you fuck",
    "homopetersymonds" : "faggot",
    "criminalwar" : "war criminal",
    "she's" : "she is",
    "we'll" : "we will", 
    "i've" : "i have",
    "we've" : "we have",
    "who's" : "who is",
    "hadn't" : "had not",
    "they've" : "they have",
    "bitchbot" : "bitch bot",
    "5h1t" : "shit",
    "5uck5" : "suck",
    "nigggers" : "nigger",
    "pennnis" : "penis",
    "pneis" : "penis",
    "couldn't" : "could not",
    "here's" : "here is",
    "hasn't" : "has not",
    "people's" : "people",
    "fggt" : "faggot",
    "faggt" : "faggot",
    "peenus" : "penis",
    "we're" : "we are",
    "u" : "you",
    "r" : "are",
    "im" : "i am",
    "you're" : "you are",
    "she'd" : "she will",
    "who'd" : "who will",
    "ur" : "your"
    
    }
    
    
    def replace_words(df):
        
        comment_list = df.comment_text.tolist()
        processed_comment_list = []
        
        for comment in comment_list:
            comment = comment.lower()
            
            words = str(comment).split()
            processed_comment = ""
            for word in words:
                if word in keys:
                    word = repl[word]
                else:
                    word = re.sub(r'.*(fuck).*', 'fuck', word)
                    word = re.sub(r'.*(bitch).*', 'bitch', word)
                    word = re.sub(r'.*(nigger).*', 'nigger', word)
                    word = re.sub(r'.*(nigga).*', 'nigger', word)
                    word = re.sub(r'.*(fag).*', 'faggot', word)
                    word = re.sub(r'(\'s)$','', word)
                    word = re.sub(r'(\'re)$', '', word)
                    word = re.sub(r'\'','',word)
                processed_comment += word + ' '
                
            processed_comment_list.append(processed_comment)
            
        return processed_comment_list
    
    keys = [i for i in repl.keys()]
    
    train['comment_text'] = replace_words(train)    
    test['comment_text'] = replace_words(test)
            
    return train, test


train,test = preproc()

test.comment_text.fillna('notextatall', inplace = True)

y_cols =  ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y_train = train[y_cols].values

nrows_train = train.shape[0]
nrows_test = test.shape[0]

def get_train_valid(X_train, y_train, test_size = 0.1):
    
    X_train, X_valid, y_train, y_valid = train_test_split(
            X_train, 
            y_train,
            test_size = test_size,
            shuffle = True
        )
    
    return X_train, X_valid, y_train, y_valid

def get_oh_features():
    """one hot features"""
    
    text = pd.concat([train.comment_text, test.comment_text], axis=0)
        
    # One Hot Features
    enc = OneHotEncoder(sparse=False)
    
    has_ipaddr = text.str.contains(r'\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}').values.reshape(-1,1)
    has_ipaddr = enc.fit_transform(has_ipaddr)
    
    has_utc = text.str.contains(r'(UTC)').values.reshape(-1,1)
    has_utc = enc.fit_transform(has_utc)

    has_url = text.str.contains(r'http[s]?://').values.reshape(-1,1)
    has_url = enc.fit_transform(has_url)
    
    has_quote = text.str.contains(r'\"').values.reshape(-1,1)
    has_quote = enc.fit_transform(has_quote)

    oh_feats = np.hstack([has_ipaddr, has_url, has_quote]) 
        
    tr_feats = oh_feats[:nrows_train]
    te_feats = oh_feats[nrows_train:]
    
    return tr_feats, te_feats

def get_cont_features():
    """continuous features"""
    text = pd.concat([train.comment_text, test.comment_text], axis=0)
    scaler = StandardScaler(with_mean = True, with_std = True)
    
    num_words = text.apply(lambda x: len( re.findall("\S+", x )))
 
    num_chars = text.apply(lambda x: len(x))    
    num_digits = text.apply(lambda x: sum(c.isdigit() for c in x))
    pct_digits = (num_digits / num_chars).reshape(-1,1)
    pct_caps = (text.str.count(r'[A-Z]').values / num_chars).reshape(-1,1)
    num_digits = scaler.fit_transform(num_digits[:, np.newaxis])
    num_chars = scaler.fit_transform(num_chars[:,np.newaxis])  
    num_words = scaler.fit_transform(num_words[:,np.newaxis])
    
    feats = np.hstack([num_digits, num_words, num_chars, pct_digits, pct_caps])
    
    tr_feats = feats[:nrows_train]
    te_feats = feats[nrows_train:]
    
    return tr_feats, te_feats



def get_glove(fpath='../input/glove.840B.300d.txt', dim=300):
    """"creates the embedding dictionary with pretrained glove embedding matrix weights"""
    
    all_comment_text = np.hstack([train.comment_text.values, test.comment_text.values])
    #print(all_comment_text[0:100])
    # converts text to sequences of integers
    tokenizer = text.Tokenizer()
    
    # get a unique integer for each word in the training set.
    tokenizer.fit_on_texts(all_comment_text)

    # The number of unique words in raw_text. 
    vocab_size = len(tokenizer.word_index) + 1
    
    embedding_vectors = {}
    f = open(fpath)
    for line in f:
        values = line.split(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_vectors[word] = coefs
    f.close()
    
    # prepare embedding matrix
    embedding_matrix = np.zeros((vocab_size, dim))
    
    missing_words = []
    
    for word, i in tokenizer.word_index.items():
        embedding_vector = embedding_vectors.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            missing_words.append(word)
            
            
    print('missing from {}: {} words'.format(fpath, len(missing_words)))


    #emb_matrix2 = _create_embedding_matrix('../input/glove.twitter.27B.200d.txt', tokenizer, 200)

    # Use a histogram to determine the max sequence length in order to limit the 
    # network size.  Used to set embed_dict['emb_input_seq_len']
    #doc_lengths = [len(doc) for doc in seq_train] 
    #plt.histogram(doc_lengths, num_bins=1)
               
    #pickle.dump(embed_dict, open(EMBED_DICT_PATH, 'wb'))
    
    return tokenizer, embedding_matrix, missing_words


#t,e,missing_words = get_glove()

def get_fasttext():
    

    '''
    all_comment_text = np.hstack([train.comment_text.values, test.comment_text.values])
    n_gram_max = 2
    
    def add_ngram(q, n_gram_max):
            ngrams = []
            for n in range(2, n_gram_max+1):
                for w_index in range(len(q)-n+1):
                    ngrams.append('--'.join(q[w_index:w_index+n]))
            return q + ngrams
        
    
    docs = []
    for doc in all_comment_text:
        doc = doc.split()
        docs.append(' '.join(add_ngram(doc, n_gram_max)))
        
    
    tok = text.Tokenizer(lower=False, filters='')
    tok.fit_on_texts(docs)
    #min_count = 2
    #num_words = sum([1 for _, v in tok.word_counts.items() if v >= min_count])
    '''
    
    all_comment_text = np.hstack([train.comment_text.values, test.comment_text.values])
    
    tok = text.Tokenizer(lower=False, filters='')
    tok.fit_on_texts(all_comment_text)
    
    embeddings_index = {}
    f = codecs.open('../input/crawl-300d-2M.vec', encoding='utf-8')
    for line in tqdm(f):
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
        
    f.close()
    print('found %s word vectors' % len(embeddings_index))
    
    ### embedding matrix
    print('preparing embedding matrix...')
    words_not_found = []
    
    embedding_matrix = np.zeros((len(tok.word_index) +1, 300))
    for word, i in tok.word_index.items():
        
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    
    print('number of null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))
    
    ###

    
    X_train = tok.texts_to_sequences(train.comment_text.values)
    X_test = tok.texts_to_sequences(test.comment_text.values)
    
    X_train = sequence.pad_sequences(X_train, maxlen = WORD_SEQ_LEN)
    X_test = sequence.pad_sequences(X_test, maxlen = WORD_SEQ_LEN)
    print(len(tok.word_index), np.max(X_train), np.max(X_test))

    return X_train, X_test, embedding_matrix

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
    

def get_keras_word_data():
    tokenizer, embedding_matrix, missing_words = get_glove()
    
    # integer encode each word in each document / sample. 
    X_train = tokenizer.texts_to_sequences(train.comment_text.values)
    X_test = tokenizer.texts_to_sequences(test.comment_text.values)
    
    # make all sequences the same length for keras by padding them with 0s
    X_train = sequence.pad_sequences(X_train, maxlen = WORD_SEQ_LEN)
    X_test = sequence.pad_sequences(X_test, maxlen = WORD_SEQ_LEN)

    return X_train, X_test, embedding_matrix, missing_words

def get_keras_char_data():
    #pat = re.compile(r'[^a-zA-Z0-9\`\~\!\@\#\$\%\^\&\*\(\)\-\+\=\[\{\]\}\'\<\,\.\>\?\/\"\;\:\_\s.*]')
    
    X_train = np.zeros((train.shape[0], CHAR_SEQ_LEN))
    X_test = np.zeros((test.shape[0], CHAR_SEQ_LEN))
    
    for i, comment in enumerate(train.comment_text.values):
        X_train[i] = nltk_utils.get_comment_ids(comment, CHAR_SEQ_LEN)
    
    for i, comment in enumerate(test.comment_text.values):
        X_test[i] = nltk_utils.get_comment_ids(comment, CHAR_SEQ_LEN)
    

    vocab_size = nltk_utils.get_vocab_size()

    return X_train, X_test, y_train, vocab_size#, embedding_matrix
    



