import numpy as np


#ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"/\\|_@#$%^&*~`+ =<>()[]{}"  # len: 69
# !"#$%&'()*+,-./0123456789:;<=>@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-,;.!?:\'\"/\\|_@#$%^&*~`+ =<>()[]{}"  


print(len(ALPHABET)) 

def get_char_dict():
    cdict = {}
    for i, c in enumerate(ALPHABET):
        cdict[c] = i + 2

    return cdict

def get_comment_ids(text, seq_len):
    
    embedding_vectors = {}
    
    with open('../input/glove.840B.300d-char.txt', 'r') as f:
        for line in f:
            line_split = line.strip().split(' ')
            char = line_split[0]
            vec = np.array(line_split[1:], dtype=float)
            embedding_vectors[char] = vec
    embedding_matrix = np.zeros((len(ALPHABET), 300))
    
    for i, char in enumerate(ALPHABET):
        print("char", char, i)
        embedding_vector = embedding_vectors.get(char)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    array = np.ones(seq_len)
    count = 0
    cdict = get_char_dict()

    for ch in text:
        if ch in cdict:
            array[count] = cdict[ch]
            count += 1

        if count >= seq_len - 1:
            return array

    return array, embedding_matrix, embedding_vectors


def get_conv_shape(conv):
    return conv.get_shape().as_list()[1:]

arr, embedding_mat, embedding_vectors = get_comment_ids('good comment', 15)