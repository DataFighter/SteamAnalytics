#!/usr/bin/env python

"""
The calling function for loading models and parameters, building models,
    and running the lstm model again some dataset.

The IdeaNet_lstm object becomes the object from which everything else is called.
First you instantiate the object, depending on the instantiation there are a series of actions:
    > Path to a parameters file and corresponding data
        >> generate model params
        >> build a model
    > Path to an existing model file
        >> train a NN
    > Path to an existing trained NN
        >> tag/classify a given training set
"""

# I need to be able to reference the other files and functions in this module
#   using os and inspect is only necessary if I'm referencing entire directories...
# import os, inspect
# cmd_folder = os.path.realpath( os.path.abspath( os.path.split( inspect.getfile( inspect.currentframe() ))[0]))
# if cmd_folder not in sys.path:
#     sys.path.insert(0, cmd_subfolder)

# from Synapsify.loadCleanly import sheets as sh

import os, sys, inspect
import numpy as np
import synapsify_preprocess as spp
import load_params as lp
# import build_model as bm
# import lstm_class as LSTM

default_dir = os.path.realpath(os.path.abspath(os.path.join(os.path.split(inspect.getfile( inspect.currentframe() ))[0],"../params")))
default_params = 'orig_params.json'

class IdeaNet(): # Is it possible to inherit any of the other classes for building these IdeaNets?

    def __init__(self, directory=default_dir, paramfile=default_params, **kwargs):
        num_args = len(kwargs.keys())
        if num_args >0:
            self._directory = kwargs['directory']
            self._filename  = kwargs['filename']
        else:
            self._directory = directory
            self._filename  = paramfile
        self._param_file = os.path.join(self._directory, self._filename)

    def __iter__(self):
        '''What would we stream? It would have to be loading something and operating on it
            This makes sense if we have an existing trained NN and want to tag some dataset
        '''

    def __next__(self):
        '''Part of __iter__'''


    #====================================================================
    # LOAD MODEL OPTIONS LOAD MODEL OPTIONS LOAD MODEL OPTIONS
    #====================================================================
    # class load_model_options():
    # We are not going to tackle inhereted classes just yet,
    #   but I can see a class that takes some set of parameters
    #   and automatically switches between sub functions based on those parameters (removing their management from the user)
    def gen_sent_tvt(self, textcol, sentcol, train_size, test_size):
        '''Generate Sentiment Test Validate Train sets from parameters
            textcol
            sentcol
            train_size
            test_size
        '''
        # directory = "/Users/dogfish/Documents/Sean/Gonzalez_Associates_LLC/Synapsify/git/IdeaNets/Synapsify_data"
        # filename = "Annotated_Comments_for_Always_Discreet_1.csv"
        self._textcol = textcol
        self._sentcol = sentcol
        self._train_size = train_size
        self._test_size  = test_size

        # Copying the imdb_preprocess.py process
        SPP = spp(self._directory, self._filename, textcol, sentcol, train_size, test_size)
        TVT = SPP.main() # assume it has 'valid' fields.

        # Model options
        self._model_options = lp.JSON_minify(self._param_file)
        print "model options: ", self._model_options

        ydim = np.max(TVT['train'][1])+1
        self._model_options['ydim'] = ydim # Depends on data used, so can't be loaded

        self._params = lp.init_params(self._model_options)
        print "model parameters: ", self._params

    def gen_tag_tvt(self):
        '''Not sure what this will entail yet, but I believe we will need to move beyond sentiment'''

    #====================================================================
    # LOAD OR BUILD MODEL LOAD OR BUILD MODEL LOAD OR BUILD MODEL
    #====================================================================
    def gen_model(self):

        if load_model:
            load_model()
        else:
            Model = build_model(self._params, self._model_options)
            lstm_model = Model.main()


    #====================================================================
    # TRAIN LSTM TRAIN LSTM TRAIN LSTM TRAIN LSTM TRAIN LSTM TRAIN LSTM
    #====================================================================
    def train(self):

        LSTM = LSTM(self._params,self._model_options)
        LSTM = LSTM.train() # are all the parameters on initialization?

    #====================================================================
    # COMPARE LSTM COMPARE LSTM COMPARE LSTM COMPARE LSTM COMPARE LSTM
    #====================================================================

    # tagged_data = LSTM(tag_this_data?)
    # calc_lstm_stats(tagged_data)
