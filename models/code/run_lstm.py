"""
Once built, load and run the lstm model again some dataset
"""

import cPickle

mo_file = 'model_options.p'
f = file(mo_file,'rb')
cPickle.load(f)

