#!/usr/bin/env python

'''
The different IdeaNet models are meant to be used as part of a larger system.
The first example of that system is 'colearning', which is meant to build
    a cooperative approach to building algorithmic 'consensus'.
'''

import os, sys, inspect

this_dir = os.path.realpath( os.path.abspath( os.path.split( inspect.getfile( inspect.currentframe() ))[0]))
lstm_dir = os.path.realpath( os.path.abspath( os.path.join( this_dir, "../models/")))
lstm_params_dir  = os.path.realpath( os.path.abspath( os.path.join( lstm_dir, "lstm/params/")))
lstm_data_dir  = os.path.realpath( os.path.abspath( os.path.join( lstm_dir, "lstm/data/")))
lstm_code_dir  = os.path.realpath( os.path.abspath( os.path.join( lstm_dir, "lstm/scode/")))
if lstm_dir not in sys.path:
    sys.path.insert(0, lstm_dir)
    sys.path.insert(0, lstm_code_dir)

from load_params import Load_LSTM_Params
from lstm_class import LSTM as lstm

param_file = 'orig_params.json'
data_file  = ''

PD = Load_LSTM_Params(lstm_params_dir, param_file)
PD.preprocess()
PD.update_options()
print PD.model_options
# Here I can pickle the PD object for use later. Good if the data is HUGE

LSTM = lstm(PD)
LSTM.build_model()
LSTM.train_model().test_model()

# IN.gen_sent_tvt(0,5,100,100)