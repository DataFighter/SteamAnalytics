#!/usr/bin/env python

'''
Testing the execution of our LSTM objects for:
    1. Loading parameters and options
    2. Building models
    3. Training nodes
    4. Tagging sets
'''

import os, sys, inspect
import IdeaNets

this_dir    = os.path.realpath( os.path.abspath( os.path.split( inspect.getfile( inspect.currentframe() ))[0]))
ideanet_dir = os.path.realpath( os.path.abspath( os.path.join( this_dir, "../../../")))
params_dir  = os.path.realpath( os.path.abspath( os.path.join( this_dir, "../params")))
if ideanet_dir not in sys.path:
    sys.path.insert(0, ideanet_dir)

paramfile = 'orig_params.json'

IN = IdeaNets.IdeaNet(params_dir, paramfile)
IN.gen_sent_tvt(0,5,100,100)