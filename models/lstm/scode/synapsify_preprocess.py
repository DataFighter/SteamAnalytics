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

import os, sys, inspect
import numpy as np

cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../../../Synapsify")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from Synapsify.loadCleanly import sheets as sh
import sys		### Should be added if we use the package "sys"
from subprocess import Popen, PIPE	### Popen and PIPE should be added as we use its functions

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_cmd = ['./tokenizer.perl', '-l', 'en', '-q', '-']
DICTIONARY = []

class synapsify_preprocess():

    def __init__(self,directory, filename, textcol, sentcol, train_size, test_size):
        self._directory  = directory
        self._filename   = filename
        self._textcol    = textcol
        self._sentcol    = sentcol
        self._train_size = train_size
        self._test_size  = test_size
        self._model_options = JSON_minify(os.path.join(directory,filename))

    def tokenize(self,sentences):

        print 'Tokenizing..',
        text = "\n".join(sentences)
        tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
        tok_text, _ = tokenizer.communicate(text)
        toks = tok_text.split('\n')[:-1]
        print 'Done'

        return toks

    def build_dict(self,sentences):

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

    def get_sentiment_indices(rows, sentcol):

        # sx = np.where(np.in1d(sent_flags, row[sentcol]))[0]
        XX = {}
        XX['pos'] = [r for r,row in enumerate(rows) if ((row[sentcol]=='Positive') | (row[sentcol]=='Neutral'))]
        XX['neg'] = [r for r,row in enumerate(rows) if ((row[sentcol]=='Negative') | (row[sentcol]=='Mixed'))]
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
        initial = len(forbidden)-1
        XX = range(initial,initial+num_indices)
        if XX[-1]>len_set: print "Test/Train set indices are out of bounds!!"
        return XX

    def main():

        # For Synapsify Core output, the comments are in the first column
        #   and the sentiment is in the 6th column
        header, rows = sh.get_spreadsheet_rows(os.path.join(self._directory, self._filename) ,self._textcol)
        sentences = [str(S[self._textcol]) for s, S in enumerate(rows)]
        len_sentences = len(sentences)
        DICTIONARY = build_dict(sentences)

        # TRAINING SET TRAINING SET TRAINING SET TRAINING SET
        train_xx = get_rand_indices(len_sentences, self._train_size,[])
        XX = get_sentiment_indices([rows[r] for r in train_xx], self._sentcol)
        train_x_sets, train_x, train_y = munge_class_freqs(sentences,[XX['neg'],XX['pos']])

        # TESTING SET TESTING SET TESTING SET TESTING SET
        test_xx = get_rand_indices(len_sentences, self._test_size,train_xx)
        XX = get_sentiment_indices([rows[r] for r in test_xx], self._sentcol)
        test_x_sets, test_x, test_y = munge_class_freqs(sentences,[XX['neg'],XX['pos']])

        TT = {
            'train_x_sets': train_x_sets,
            'train_x': train_x,
            'train_y': train_y,
            'test_x_sets': test_x_sets,
            'test_x': test_x,
            'test_y': test_y
        }

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
