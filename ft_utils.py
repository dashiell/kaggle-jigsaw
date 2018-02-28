FASTTEXT_PATH = '/home/computer/bin/fasttext'
MODEL_PATH = '../input/all_glove.skipgram.bin'
NUM_NEIGHBORS = 10

import pexpect

class NNLookup:
    """Class for using the command-line interface to fasttext nn to lookup neighbours.
    It's rather fiddly and depends on exact text strings. But it is at least short and simple."""
    def __init__(self):
        self.nn_process = pexpect.spawn('%s nn %s %d' % (FASTTEXT_PATH, MODEL_PATH, NUM_NEIGHBORS))
        self.nn_process.expect('Query word?')  # Flush the first prompt out.

    def get_nn(self, word):
        self.nn_process.sendline(word)
        self.nn_process.expect('Query word?')
        output = self.nn_process.before
        output = output.decode('utf-8')
        return [word] + [line.strip().split()[0] for line in output.strip().split('\n')[1:]]
    
ft = NNLookup()

test = ft.get_nn('niger')
print(test)


####
'''
# first need to do this to create fasttext skipgrams:
## fasttext skipgram -input all_glove.txt -output all_glove.skipgram

out_words = []

fpath = '../input/glove.840B.300d.txt'

f = open(fpath)
for line in f:
    values = line.split(' ')
    word = values[0] + '\n'
    out_words.append(word)
f.close()

f = open('../input/all_glove.txt', 'w')
#out_words = map(lambda x: x+'\n', out_words)
for word in out_words:
    f.write(word)
f.close()
''' 

