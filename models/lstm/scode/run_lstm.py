"""
The calling function for loading models and parameters, building models,
    and running the lstm model again some dataset.
"""

import cPickle

#====================================================================
# LOAD MODEL OPTIONS LOAD MODEL OPTIONS LOAD MODEL OPTIONS
#====================================================================
mo_file = 'model_options.p'
f = file(mo_file,'rb')
cPickle.load(f)

synapsify_preprocess()

params, model_options = load_params(param_file)

build_model(params, model_options)

#====================================================================
# LOAD OR BUILD MODEL LOAD OR BUILD MODEL LOAD OR BUILD MODEL
#====================================================================

load_model()


#====================================================================
# TRAIN LSTM TRAIN LSTM TRAIN LSTM TRAIN LSTM TRAIN LSTM TRAIN LSTM
#====================================================================

train_lstm()