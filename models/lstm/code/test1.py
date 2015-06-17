import theano
import theano.tensor as T
from collections import OrderedDict
import numpy

params = OrderedDict()	### params is a python ordered dictionary.
r = numpy.random.rand(10, 5)	### Generate a random matrix with dimension of 10 * 5.
params['haha'] = (0.01 * r).astype('float32')	### insert a key 'haha' with its value into the dictionary.

sparams = OrderedDict() ### sparams is another python ordered dictionary.
sparams['haha'] = theano.shared(params['haha'], name='haha')	### Create Theano Shared Variable from the params.

x = T.matrix('x', dtype='int64') ### x is a theano tensor matrix

n1 = x.shape[0] 	### What does it mean?
n2 = x.shape[1]		### What does it mean?
 
h = sparams['haha'][x.flatten()].reshape([n1, n2, 5])	### What does it mean?

