#!/usr/bin/env python

"""
These spark utilities are the beginning of an IdeaNets Spark Class (INS).
The INS will help people to leverage their Deep Learning clusters (AWS, NVIDIA Torch X, etc.)
The INS will automatically call the different models requested (passed or referenced) and map-reduce them for the user.
"""

import numpy as np
from pyspark import SparkConf, SparkContext
import os, sys, inspect
import os.path.join as opj
import uuid

this_path = os.path.abspath(os.path.dirname(__file__))

class INSpark():

    def __init__(self, app_name=None, **kwargs):
        if app_name==None:
            uid = uuid.uuid1()
        else:
            uid = app_name

        user_params = {}
        for key in kwargs:
            user_params[key] = kwargs[key]

        self.app = SparkConf().setAppName(uid)

        return self

    @classmethod
    def initialize(self, url="local", data_dir=None, functions=None, user_params="synapsify_data"):
        ''' Create the connection to the cluster (e.g. AWS) and prepare to execute
        :return:
        '''

        ### SET THE MASTER NODE-------------------------------------------
        self._conf = self.app.setMaster(url)

        ### USE DEFAULT DATA FOR TESTING----------------------------------
        if data_dir==None: # Assume IdeaNet testing directory
            print "Using IdeaNet testing files"
            data_dir = opj(this_path,"../data")

        ### USE LSTM MODEL IF NONE IS SPECIFIED---------------------------
        if functions==None:
            models = [opj(this_path,"../models/lstm/scode/lstm_class.py")]
        else:
            models = []
            for f,func in enumerate(functions):
                models.append(opj(this_path,func))

        ### Setup the Spark configuration.
        sc = SparkContext(conf = self._conf, pyFiles=models)

        ### ORGANIZE THE PARAMETERS FOR EACH SPARK INSTANCE
        spark_lstm = []
        for param in user_params:
            del LSTM
            LSTM = lstm(param_file=param)
            LSTM.preprocess()
            spark_lstm.append(LSTM)

        self.dist_lstm = sc.parallelize(spark_lstm)

        self.dist_lstm.spawn(slave_urls)

    def run(self):

        self.dist_lstm.map(lambda x: x.build_model().train_model().test_model())

    def reduce(self):

        self.dist_lstm.reduce("reduce to somewhere...")