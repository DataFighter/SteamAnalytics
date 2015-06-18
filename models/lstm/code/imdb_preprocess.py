"""
This script is what created the dataset pickled.
1) You need to download this file and put it in the same directory as this file.
https://github.com/moses-smt/mosesdecoder/raw/master/scripts/tokenizer/tokenizer.perl . Give it execution permission.
2) Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/ and extract it in the current directory.
3) Then run this script.
"""


import numpy
import cPickle as pkl
import os, sys, inspect

from collections import OrderedDict

import glob
import os

from subprocess import Popen, PIPE

dataset_path = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"aclImdb/")))
if dataset_path not in sys.path:
    sys.path.insert(0, dataset_path)

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_perl = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"tokenizer.perl")))
tokenizer_cmd = [tokenizer_perl, '-l', 'en', '-q', '-']


def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)

    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    ### print toks
    return toks


def build_dict(path):
    sentences = []
    currdir = os.getcwd()
    os.chdir('%s/pos/' % path)

    ### Read raw data from the dataset.
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir('%s/neg/' % path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)

    ### "sentences" is a long list containing all sentences
    sentences = tokenize(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:	### Get the appearing frequency of each word.
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()	### "counts" is the value list.
    keys = wordcount.keys()	### "keys" is the key list conataining all words.

    sorted_idx = numpy.argsort(counts)[::-1]	### sort "counts" in the list and return the index list.

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):	### The dictionary data format: key: word, value: order index (The order is descending according to the word apprearing frequency.)
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)	### The order index starts from 2.

    print numpy.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def grab_data(path, dictionary):
    sentences = []
    currdir = os.getcwd()
    os.chdir(path)
    for ff in glob.glob("*.txt"):
        with open(ff, 'r') as f:
            sentences.append(f.readline().strip())
    os.chdir(currdir)
    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)

    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [dictionary[w] if w in dictionary else 1 for w in words]	

    return seqs		### returns the order index of each word.


def main():
    # Get the dataset from http://ai.stanford.edu/~amaas/data/sentiment/

    path = dataset_path
    dictionary = build_dict(os.path.join(path, 'train'))

    train_x_pos = grab_data(os.path.join(path, 'train/pos'), dictionary)
    train_x_neg = grab_data(os.path.join(path, 'train/neg'), dictionary)
    train_x = train_x_pos + train_x_neg
    train_y = [1] * len(train_x_pos) + [0] * len(train_x_neg)

    test_x_pos = grab_data(os.path.join(path, 'train/pos'), dictionary)
    test_x_neg = grab_data(os.path.join(path, 'train/neg'), dictionary)
    test_x = test_x_pos + test_x_neg
    test_y = [1] * len(test_x_pos) + [0] * len(test_x_neg)

    f = open('imdb.pkl', 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open('imdb.dict.pkl', 'wb')
    pkl.dump(dictionary, f, -1)
    f.close()

if __name__ == '__main__':
    main()
