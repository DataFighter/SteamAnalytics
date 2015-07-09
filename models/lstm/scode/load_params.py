#!/usr/bin/env python

import ast, sys, os, inspect
import numpy as np

this_dir = this_dir = os.path.realpath( os.path.abspath( os.path.split( inspect.getfile( inspect.currentframe() ))[0]))
params_directory = os.path.realpath(os.path.abspath(os.path.join(this_dir,"../params")))
if this_dir not in sys.path: sys.path.insert(0, this_dir)
if params_directory not in sys.path: sys.path.insert(0, params_directory)

from synapsify_preprocess import Preprocess

class Load_LSTM_Params(Preprocess):

    def __init__(self, params_dir=params_directory, param_file='ruofan_params.json'):
        self._params_dir = params_dir
        self._param_file = param_file
        self.model_options = self.JSON_minify(os.path.join(self._params_dir, self._param_file))
        Preprocess.__init__(self, self.model_options)

    @classmethod
    def JSON_minify(self, filename): # assumes it's in this directory

        # f = open('../params/'+filename,'r')
        f = open(filename,'r')

        json = ''
        eof = False
        while eof==False:
            line = f.readline()
            tmp_line = line
            if tmp_line==[]:
                eof==True; break
            ex = tmp_line.find('\n')
            if ex==-1: # eof
                ex = len(tmp_line)
                eof = True
            cx = tmp_line.find('#')
            if cx==-1:
                json += tmp_line[0:ex]
            else:
                json += tmp_line[0:(cx-1)]

        dict = ast.literal_eval(json)
        return dict

    #==============================================================================
    # imdb.py functions imdb.py functions imdb.py functions imdb.py functions
    #==============================================================================
    @classmethod
    def _get_dataset_file(self, dataset, default_dataset, origin):
        '''Look for it as if it was a full path, if not, try local file,
        if not try in the data directory.

        Download dataset if it is not present

        '''
        data_dir, data_file = os.path.split(dataset)
        if data_dir == "" and not os.path.isfile(dataset):
            # Check if dataset is in the data directory.
            new_path = os.path.join(
                os.path.split(__file__)[0],
                "..",
                "data",
                dataset
            )
            if os.path.isfile(new_path) or data_file == default_dataset:
                dataset = new_path

        if (not os.path.isfile(dataset)) and data_file == default_dataset:
            import urllib
            print 'Downloading data from %s' % origin
            urllib.urlretrieve(origin, dataset)
        return dataset

    @classmethod
    def _load_raw_data(self, path="imdb.pkl"):
        ''' I just want to know the file structure so I can train my own model!!'''

        # Load the dataset
        path = get_dataset_file(
            path, "imdb.pkl",
            "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl")

        if path.endswith(".gz"):
            f = gzip.open(path, 'rb')
        else:
            f = open(path, 'rb')

        train_set = cPickle.load(f)
        test_set = cPickle.load(f)

        return train_set, test_set
    #==============================================================================
    # imdb.py functions imdb.py functions imdb.py functions imdb.py functions
    #==============================================================================
    # From lstm.py From lstm.py From lstm.py From lstm.py From lstm.py From lstm.py
    #==============================================================================

    @classmethod
    def _load_params(self, path, params):
        pp = np.load(path)
        for kk, vv in params.iteritems():
            if kk not in pp:
                raise Warning('%s is not in the archive' % kk)
            params[kk] = pp[kk]

        return params

    #==============================================================================
    # From lstm.py From lstm.py From lstm.py From lstm.py From lstm.py From lstm.py
    #==============================================================================

    @classmethod
    def update_options(self):

        # Assemble directories
        # param_dir_file = os.path.join(self._params_dir, self._param_file)
        #
        # # Model options
        # self.model_options = self.JSON_minify(param_dir_file)
        # print "model options", self.model_options
        #
        # SP = sp.synapsify_preprocess(self.model_options)
        # SP.load()

        # print 'Loading data'
        # train, valid, test = load_data(n_words       = model_options['n_words'],
        #                                valid_portion = model_options['0.05'],
        #                                maxlen        = model_options['maxlen'])

        ydim = np.max(self.train_set[1])+1

        self.model_options['ydim'] = ydim

        # THIS IS INELEGANT THIS IS INELEGANT THIS IS INELEGANT
        # self.train_set = SP.train_set
        # self.valid_set = SP.valid_set
        # self.test_set  = SP.test_set

        return self
        # return params, model_options, TVT


if __name__ == '__main__':
    filename = sys.argv[1]
    main(filename)