"""
This file is designed to ingest Synapsify standard tagged data sets and convert them to LSTM input format

Input:
    1. Directory and filename of tagged dataset to be converted
    2. Token dictionary - text file where each row is a new word

Output:
    LSTM intput file structure - 2xN array
        Columns:
            2x1 vector
        Rows:
            1st row: vector of indices to token dictionary
            2nd row: total sentiment of that vector
"""

"""
This code is used for generating a single ".pkl" file storing both training and testing dataset, and the parameters are passed from script lines.
"""

import os, sys, inspect
import cPickle as pkl
import numpy as np

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../Synapsify")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from Synapsify.loadCleanly import sheets as sh
from subprocess import Popen, PIPE

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']
DICTIONARY = []


def tokenize(sentences):

    print 'Tokenizing..',
    text = "\n".join(sentences)
    tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
    tok_text, _ = tokenizer.communicate(text)
    toks = tok_text.split('\n')[:-1]
    print 'Done'

    return toks

def build_dict(sentences):

    sentences = tokenize(sentences)

    print 'Building dictionary..',
    wordcount = dict()
    for ss in sentences:
        words = ss.strip().lower().split()
        for w in words:
            if w not in wordcount:
                wordcount[w] = 1
            else:
                wordcount[w] += 1

    counts = wordcount.values()
    keys = wordcount.keys()

    sorted_idx = np.argsort(counts)[::-1]

    worddict = dict()

    for idx, ss in enumerate(sorted_idx):
        worddict[keys[ss]] = idx+2  # leave 0 and 1 (UNK)

    print np.sum(counts), ' total words ', len(keys), ' unique words'

    return worddict


def format_sentence_frequencies(sentences):

    sentences = tokenize(sentences)

    seqs = [None] * len(sentences)
    for idx, ss in enumerate(sentences):
        words = ss.strip().lower().split()
        seqs[idx] = [DICTIONARY[w] if w in DICTIONARY else 1 for w in words]

    return seqs

def get_sentiment_indices(rows, sentcol, init):

    XX = {}
    XX['pos'] = [r + len(init) for r,row in enumerate(rows) if ((row[sentcol]=='Positive') or (row[sentcol]=='Neutral'))]
    XX['neg'] = [r + len(init) for r,row in enumerate(rows) if ((row[sentcol]=='Negative') or (row[sentcol]=='Mixed'))]
    return XX

def munge_class_freqs(sentences,index_sets):

    # A variation on the original LSTM code,
    freqs_x_sets = []
    freqs_x = []
    freqs_y = []
    for y,xx in enumerate(index_sets):
        x_set = format_sentence_frequencies([sentences[x] for x in xx])
        freqs_x_sets.append( x_set)
        freqs_x = freqs_x + x_set
        freqs_y = freqs_y + [y]*len(x_set)

    return freqs_x_sets, freqs_x, freqs_y

def get_rand_indices(len_set, num_indices, forbidden):
    """
    Function is designed to extract test or training set indices
    :param len_set:
    :param num_indices:
    :param forbidden:
    :return:
    """

    # I just want to get this working and move on
    initial = len(forbidden)  ### Should from 0
    XX = range(initial,initial+num_indices)
    if XX[-1]>len_set: print "Test/Train set indices are out of bounds!!"
    return XX

def main(directory, filename, textcol, sentcol, train_size, test_size):

    # For Synapsify Core output, the comments are in the first column
    #   and the sentiment is in the 6th column
    header, rows = sh.get_spreadsheet_rows(os.path.join(directory, filename) ,textcol)
    sentences = [str(S[textcol]) for s, S in enumerate(rows)]
    len_sentences = len(sentences)  ### This part has no bugs
    global DICTIONARY
    DICTIONARY = build_dict(sentences)

    # TRAINING SET TRAINING SET TRAINING SET TRAINING SET
    train_xx = get_rand_indices(len_sentences, train_size,[])
    XX = get_sentiment_indices([rows[r] for r in train_xx], sentcol, [])
    train_x_sets, train_x, train_y = munge_class_freqs(sentences,[XX['neg'],XX['pos']])

    # TESTING SET TESTING SET TESTING SET TESTING SET
    test_xx = get_rand_indices(len_sentences, test_size,train_xx)
    XX = get_sentiment_indices([rows[r] for r in test_xx], sentcol, train_xx)
    test_x_sets, test_x, test_y = munge_class_freqs(sentences,[XX['neg'],XX['pos']])

    TT = {
        'train_x_sets': train_x_sets,
        'train_x': train_x,
        'train_y': train_y,
        'test_x_sets': test_x_sets,
        'test_x': test_x,
        'test_y': test_y
    }


    ### param name is the data file name without ".csv" which is passed into the main function.
    param_name = sys.argv[2];
    param_name = param_name.strip('.csv')   ### Delete the ".csv" part

    ### data_file_name is the file name of data set in .pkl format.
    data_file_name = param_name
    data_file_name += ".pkl"

    ### dic name is the name of the data dictionary
    dic_name = param_name
    dic_name += ".dict.pkl"
    print data_file_name, dic_name

    f = open(data_file_name, 'wb')
    pkl.dump((train_x, train_y), f, -1)
    pkl.dump((test_x, test_y), f, -1)
    f.close()

    f = open(dic_name, 'wb')
    pkl.dump(DICTIONARY, f, -1)
    f.close()

    return TT

if __name__ == '__main__':
    directory = sys.argv[1]
    filename  = sys.argv[2]
    textcol = 0
    if len(sys.argv)>2: textcol = int(sys.argv[3])
    sentcol = 5
    if len(sys.argv)>3: sentcol = int(sys.argv[4])

    ### I added the following part of codes here
    train_size = int(sys.argv[5])
    test_size = int(sys.argv[6])
    ### The main function should take six parameters instead of four
    ### The original code is:
    ### main(directory, filename, textcol, sentcol)
    main(directory, filename, textcol, sentcol, train_size, test_size)
