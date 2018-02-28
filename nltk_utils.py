import numpy as np


CHARS = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:\'\"/\\|_@#$%^&*~`+ =<>()[]{}"  # len: 69
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-,;.!?:\'\"/\\|_@#$%^&*~`+ =<>()[]{}"  # len: 69
# !"#$%&()*+,-./0123456789:;<=>@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}


def get_vocab_size():
    return len(CHARS) + 1
 

def get_char_dict():
    cdict = {}
    for i, c in enumerate(CHARS):
        cdict[c] = i + 1

    return cdict

def get_comment_ids(text, seq_len):
    
    array = np.zeros(seq_len)
    count = 0
    cdict = get_char_dict()

    for ch in text:
        if ch in cdict:
            array[count] = cdict[ch]
            count += 1

        if count >= seq_len - 1:
            return array

    return array
    