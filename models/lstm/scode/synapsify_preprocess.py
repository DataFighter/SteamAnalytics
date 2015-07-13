#!/usr/bin/env python

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

# If IdeaNets are treated as a module, this addition should not be necessary.
this_dir = os.path.split(inspect.getfile( inspect.currentframe() ))[0]
cmd_subfolder = os.path.realpath(os.path.abspath(os.path.join(this_dir,"../../../../Synapsify")))
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)

from Synapsify.loadCleanly import sheets as sh
import sys		### Should be added if we use the package "sys"
from subprocess import Popen, PIPE	### Popen and PIPE should be added as we use its functions

# tokenizer.perl is from Moses: https://github.com/moses-smt/mosesdecoder/tree/master/scripts/tokenizer
tokenizer_dir_file = os.path.realpath(os.path.abspath(os.path.join(this_dir,'./tokenizer.perl')))
tokenizer_cmd = [tokenizer_dir_file, '-l', 'en', '-q', '-']

class Preprocess():

    def __init__(self, model_options, data_file=None, text_col=None, label_col=None, train_size=None, test_size=None):
        self._data_directory  = os.path.realpath(os.path.abspath(os.path.join(this_dir,model_options['data_directory'])))

        if data_file==None:  self._data_file  = model_options['data_file'];  else: self._data_file = data_file;
        if text_col==None:   self._text_col   = model_options['text_col'];   else: self._text_col = text_col;
        if label_col==None:  self._label_col  = model_options['label_col'];  else: self._label_col = label_col;
        if train_size==None: self._train_size = model_options['train_size']; else: self._train_size = train_size;
        if test_size==None:  self._test_size  = model_options['test_size'];  else: self._test_size = test_size;

        self._n_words    = model_options['n_words']
        self._model_options = model_options #JSON_minify(os.path.join(directory,filename))
        self._DICTIONARY = []

    @staticmethod
    def _tokenize(sentences):

        print 'Tokenizing..',
        text = "\n".join(sentences)
        tokenizer = Popen(tokenizer_cmd, stdin=PIPE, stdout=PIPE)
        tok_text, _ = tokenizer.communicate(text)
        toks = tok_text.split('\n')[:-1]
        print 'Done'

        return toks

    @classmethod
    def _build_dict(self, sentences):

        sentences = self._tokenize(sentences)

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

    # @classmethod
    # def _format_sentence_freq(self,sentence):
    #     words = sentence.strip().lower().split()
    #     seq = [self._DICTIONARY[w] if w in self._DICTIONARY else 1 for w in words]
    #     return seq

    @classmethod
    def _format_sentence_frequencies(self,sentences):

        sentences = self._tokenize(sentences)

        seqs = [None] * len(sentences)
        for idx, ss in enumerate(sentences):
            # seqs[idx] = self._format_sentence_freq(ss)
            words = ss.strip().lower().split()
            seqs[idx] = [self._DICTIONARY[w] if w in self._DICTIONARY else 1 for w in words]

        return seqs

    @classmethod
    def _get_sentiment_indices(self,rows, sentcol, init):

        # sx = np.where(np.in1d(sent_flags, row[sentcol]))[0]
        XX = {}
        len_init = len(init) # Ruofan fix
        XX['pos'] = [r + len_init for r,row in enumerate(rows) if ((row[sentcol]=='Positive') | (row[sentcol]=='Neutral'))]
        XX['neg'] = [r + len_init for r,row in enumerate(rows) if ((row[sentcol]=='Negative') | (row[sentcol]=='Mixed'))]
        return XX

    @classmethod
    def _munge_class_freqs(self,sentences,index_sets):

        # A variation on the original LSTM code,
        # freqs_x_sets = []
        freqs_x = []
        freqs_y = []
        for y,xx in enumerate(index_sets):
            x_set = self._format_sentence_frequencies([sentences[x] for x in xx])
            # freqs_x_sets.append( x_set)
            freqs_x += x_set
            freqs_y += [y]*len(x_set)

        return (freqs_x, freqs_y)

    @classmethod
    def _get_rand_indices(self,len_set, num_indices, forbidden):
        """
        Function is designed to extract test or training set indices
        :param len_set:
        :param num_indices:
        :param forbidden:
        :return:
        """

        # I just want to get this working and move on
        initial = len(forbidden) # Ruofan fix
        XX = range(initial,initial+num_indices)
        if XX[-1]>len_set: print "Test/Train set indices are out of bounds!!"
        return XX

    def max_sentence_length(self):
        '''From imdb.py'''
        # from m
        # train_set = self.train_xx ?????
        maxlen = self._model_options['maxlen']
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    @classmethod
    def _split_train_w_valid_set(self, train_set):
        '''From imdb.py'''
        valid_portion = self._model_options['valid_portion']
        train_set_x, train_set_y = train_set	### trian_set_x means all attributes of each instance, and train_set_y means all labels for each instance.
        n_samples = len(train_set_x)
        sidx = np.random.permutation(n_samples)	### Shuffle the index of the training dataset
        n_train = int(np.round(n_samples * (1. - valid_portion)))	### means the number of train data set.
        ### Partition the train dataset into two parts: the train set and the validation set.
        valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
        valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
        train_set_x = [train_set_x[s] for s in sidx[:n_train]]
        train_set_y = [train_set_y[s] for s in sidx[:n_train]]

        return valid_set_x, valid_set_y, train_set_x, train_set_y

    @classmethod
    def _remove_unk(self,x):
        '''Set the value of word who is not in the dictionary to 1.'''
        return [[1 if w >= self._n_words else w for w in sen] for sen in x]

    @classmethod
    def preprocess(self):

        # For Synapsify Core output, the comments are in the first column
        #   and the sentiment is in the 6th column
        header, rows = sh.get_spreadsheet_rows(os.path.join(self._data_directory, self._data_file) ,self._text_col)
        sentences = [str(S[self._text_col]) for s, S in enumerate(rows)]
        len_sentences = len(sentences)
        self._DICTIONARY = self._build_dict(sentences)

        # Randomly split train and test data
        self._train_xx = self._get_rand_indices(len_sentences, self._train_size,[])
        self._test_xx  = self._get_rand_indices(len_sentences, self._test_size,self._train_xx)
        # max_sentence_length(self.train_x_sets) #
        # max_sentence_length(self.test_x_sets) # IS THERE A MAXIMUM LENGTH???????

        # Grab the indices for the Core sentiment
        trXX = self._get_sentiment_indices([rows[r] for r in self._train_xx], self._label_col, [])
        teXX = self._get_sentiment_indices([rows[r] for r in self._test_xx], self._label_col, self._train_xx)

        # Munge training and test sets for the classes provided
        train = self._munge_class_freqs(sentences,[trXX['neg'],trXX['pos']])
        test  = self._munge_class_freqs(sentences,[teXX['neg'],teXX['pos']])

        # Split training into a validation set per the model parameter
        valid_set_x, valid_set_y, train_set_x, train_set_y = self._split_train_w_valid_set( train)

        # Remove unknown words
        train_set_x = self._remove_unk(train_set_x)
        valid_set_x = self._remove_unk(valid_set_x)
        test_set_x  = self._remove_unk(test[0])

        self.train_set = (train_set_x, train_set_y)
        self.valid_set = (valid_set_x, valid_set_y)
        self.test_set  = (test_set_x, test[1])

        # TVT = {
        #     'train': (train_set_x, train_set_y),
        #     'valid': (valid_set_x, valid_set_y),
        #     'test': (test_set_x,test[1])
        # }

        return self

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
    load(directory, filename, textcol, sentcol, train_size, test_size)
