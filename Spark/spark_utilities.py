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
import ast
import ujson as json

this_path = os.path.abspath(os.path.dirname(__file__))

class INSpark():

    default_params = {
        "Spark_instances":"train_size"
    }

    def __init__(self, app_name=None, **kwargs):
        if app_name==None:
            uid = uuid.uuid1()
        else:
            uid = app_name

        self.app = SparkConf().setAppName(uid)

        return self

    @classmethod
    def init_model_experiment(self, url="local", data_dir=None, data_file_root=None, model="lstm", spark_params=default_params,**kwargs):
        ''' Create the connection to the cluster (e.g. AWS) and prepare to execute
        :return:
        '''

        ### SET THE MASTER NODE-------------------------------------------
        self._conf = self.app.setMaster(url)

        ### USE DEFAULT DATA FOR TESTING----------------------------------
        if data_dir==None: # Assume IdeaNet testing directory
            print "Using IdeaNet testing files"
            data_dir = opj(this_path,"../data")
        if data_file==None:
            data_file = "Annotated_Comments_for_Naytev_Facebook.csv" # how do I handle multiple files vs multiple cuts of the same file?

        ### Setup the Spark configuration.
        #  IdeaNets has a specific structure, where every model has it's own directory of the same name
        #   as well as folder scode (i.e. Synapsify code)
        if "colearn" not in model:
            model_path = opj(this_path,"../models/"+model+"/scode"+model+"_class.py")
        ast.literal_eval("import " + model_path + " as model_class")
        sc = SparkContext(conf = self._conf, pyFiles=model_path)

        ### ORGANIZE THE PARAMETERS FOR EACH SPARK INSTANCE
        # It is assumed that the spark params will mimic the model params,
        #   except each model variable will have 1 or N instances for each independent Spark instance
        # First, determine how many individual Sparks instances:
        #   we can be given a number or a key to the Spark params json:

        # Loop through kwargs and replace as the user requested
        for key, value in kwargs.iteritems():
            try:
                spark_params[key] = value
            except:
                print "Could not add: " + str(key) + " --> " + str(value)

        if isinstance( spark_params['Spark_instances'], int ):
            num_experiments = spark_params['Spark_instances']
        else: # Assume it's referencing another variable in the json input
            key = spark_params['Spark_instances']
            num_experiments = len(spark_params[key])

        ### ORGANIZE MODEL EXPERIMENTS VIA SPARK PARAMETERS-------------------------------
        spark_models = []
        self._model = model.lower()
        for e in xrange(num_experiments):
            # if model.lower()=="lstm":

            model_params = {}
            for key, value in spark_params.iteritems():
                if "spark" not in key.lower(): # Don't want to accidently pass park parameters to the model
                    if len(value)>1:
                        model_params[key] = value[e]
                    else:
                        model_params[key] = value

                model_params['data_directory'] = data_dir
                model_params['data_file'] =
            model_obj = model_class(params=lstm_params)
            spark_models.append(model_obj)
            # else:
            #     print "Your model needs to be supported."

        self.dist_lstm = sc.parallelize(spark_models)

        # self.dist_lstm.spawn(slave_urls)

    def run(self):

        if self._model=="lstm":
            self.dist_lstm.map(lambda x: x.build_model().train_model().test_model())
        else:
            print "Your model is not supported."

    def reduce(self):

        self.dist_lstm.reduce("reduce to somewhere...")