#!/usr/bin/env python

import theano, time, copy
import theano.tensor as tensor
import uuid
from theano import config
import numpy as np
import cPickle as pkl
import random
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import os, sys, inspect
this_dir = this_dir = os.path.realpath( os.path.abspath( os.path.split( inspect.getfile( inspect.currentframe() ))[0]))
params_directory = os.path.realpath(os.path.abspath(os.path.join(this_dir,"../params")))
if this_dir not in sys.path: sys.path.insert(0, this_dir)
if params_directory not in sys.path: sys.path.insert(0, params_directory)

from load_params import Load_LSTM_Params

class LSTM(Load_LSTM_Params):

    def __init__(self, Object=None, params=None):

        default_params = {
            "dim_proj":128,
            "patience":10,
            "max_epochs":5000,
            "dispFreq":10,
            "decay_c":0.,
            "lrate":0.0001,
            "n_words":10000,
            "optimizer":"adadelta",
            "encoder":"lstm",
            "saveto":"lstm_model.npz",
            "validFreq":370,
            "saveFreq":1110,
            "maxlen":100,
            "batch_size":16,
            "valid_batch_size":64,
            "valid_portion":0.05,
            "dataset":"imdb",
            "noise_std":0.,
            "use_dropout":True,
            "reload_model":"",
            "text_col":0,
            "dedupe":True,
            "label_col":5,
            "train_max":0.5,
            "train_size":1524,
            "test_size":1533,
            #"data_directory":"../data/test",
            "data_directory":"/home/ying/Deep_Learning/Synapsify_data",
            "data_file":"Annotated_Comments_for_Always_Discreet_1.csv",
            "raw_rows":None,
            "class_type":"Sentiment"
        }

        self._del_keys = ['_layers','f_grad_shared','f_grad'] #,'train_set']

        if params!=None:
            #print params
            for key,value in params.iteritems():
                try:
                    default_params[key] = value
                except:
                    print "Could not add: " + str(key) +" --> "+ str(value)
        self._params = default_params

        ### THIS OBJECT THING NEEDS TO BE SIMPLIFIED NOW THAT EVERYTHING IS INHERITED.
        if Object==None:
            Load_LSTM_Params.__init__(self, self._params)

            self._layers = {'lstm': (self.param_init_lstm, self.lstm_layer)}

        elif Object!=None: # Copy the LSTM Object

            # Params & Data variables
            self._params  = copy.deepcopy(Object._params)
            self.train_set = copy.deepcopy(Object.train_set)
            self.valid_set = copy.deepcopy(Object.valid_set)
            self.test_set  = copy.deepcopy(Object.test_set)
            self._DICTIONARY = copy.deepcopy(Object._DICTIONARY)

            try:
                print "Assuming object is LSTM, copying..."
                # LSTM variables
                self._layers = copy.deepcopy(Object._layers)
                self.model_options = copy.deepcopy(Object.model_options)
                self._params  = copy.deepcopy(Object._params)
                self._tparams = copy.deepcopy(Object._tparams) # I don't know if this will work, do Theano variables need to be recompiled?
                self.model_options = copy.deepcopy(Object.model_options)
                self.optimizer = self.model_options['optimizer']

            except:
                print "Couldn't copy LSTM Object, initializing a new object."
                self._layers = {'lstm': (self.param_init_lstm, self.lstm_layer)}
                self.model_options = Object.model_options
                self._params  = self._init_params(self.model_options)
                self._tparams = self._init_tparams(self._params)
                self.optimizer = self.model_options['optimizer']

    # @classmethod
    def _init_params(self, options):
        """
        Global (not LSTM) parameter. For the embeding and the classifier.
        """
        params = OrderedDict()
        # embedding
        randn = np.random.rand(options['n_words'], options['dim_proj'])
        params['Wemb'] = (0.01 * randn).astype(config.floatX) #.astype('float32')
        params = self.get_layer(options['encoder'])[0](options,
                                                  params,
                                                  prefix=options['encoder'])
        # classifier
        params['U'] = 0.01 * np.random.randn(options['dim_proj'],
                                             options['ydim']).astype(config.floatX) #.astype('float32')
        params['b'] = np.zeros((options['ydim'],)).astype(config.floatX) #.astype('float32')

        return params

    @staticmethod
    def _init_tparams( params):
        tparams = OrderedDict()
        for kk, pp in params.iteritems():
            tparams[kk] = theano.shared(params[kk], name=kk)
        return tparams

#=======================================================================================================================
# MODEL BUILDING MODEL BUILDING MODEL BUILDING MODEL BUILDING MODEL BUILDING MODEL BUILDING MODEL BUILDING MODEL BUILDING
#=======================================================================================================================

    @staticmethod
    def _p(pp, name):
        return '%s_%s' % (pp, name)

    @staticmethod
    def numpy_floatX(data):
        return np.asarray(data, dtype=config.floatX)

    @staticmethod
    def ortho_weight(ndim):
        W = np.random.randn(ndim, ndim)
        u, s, v = np.linalg.svd(W)
        return u.astype(config.floatX)

    @staticmethod
    def dropout_layer( state_before, use_noise, trng):
        proj = tensor.switch(use_noise,
                             (state_before *
                              trng.binomial(state_before.shape,
                                            p=0.5, n=1,
                                            dtype=state_before.dtype)),
                             state_before * 0.5)
        return proj

    # @classmethod
    def lstm_layer(self, tparams, state_below, options, prefix='lstm', mask=None):

        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1

        assert mask is not None

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = tensor.dot(h_, tparams[self._p(prefix, 'U')])
            preact += x_
            preact += tparams[self._p(prefix, 'b')]

            i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
            f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
            o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
            c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

        state_below = (tensor.dot(state_below, tparams[self._p(prefix, 'W')]) +
                       tparams[self._p(prefix, 'b')])

        dim_proj = options['dim_proj']
        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[tensor.alloc(self.numpy_floatX(0.),
                                                               n_samples,
                                                               dim_proj),
                                                  tensor.alloc(self.numpy_floatX(0.),
                                                               n_samples,
                                                               dim_proj)],
                                    name=self._p(prefix, '_layers'),
                                    n_steps=nsteps)
        return rval[0]

    # @classmethod
    def get_layer(self, name):
        fns = self._layers[name]
        return fns

    # @classmethod
    def param_init_lstm(self, options, params, prefix='lstm'):
        """
        Init the LSTM parameter:
        :see: init_params
        """

        W = np.concatenate([self.ortho_weight(options['dim_proj']),
                            self.ortho_weight(options['dim_proj']),
                            self.ortho_weight(options['dim_proj']),
                            self.ortho_weight(options['dim_proj'])], axis=1)
        params[self._p(prefix, 'W')] = W
        U = np.concatenate([self.ortho_weight(options['dim_proj']),
                            self.ortho_weight(options['dim_proj']),
                            self.ortho_weight(options['dim_proj']),
                            self.ortho_weight(options['dim_proj'])], axis=1)
        params[self._p(prefix, 'U')] = U
        b = np.zeros((4 * options['dim_proj'],))
        params[self._p(prefix, 'b')] = b.astype(config.floatX)

        return params

    # @classmethod
    def adadelta(self, lr, tparams, grads, x, mask, y, cost):

        zipped_grads = [theano.shared(p.get_value() * self.numpy_floatX(0.),
                                      name='%s_grad' % k)
                        for k, p in tparams.iteritems()]
        running_up2 = [theano.shared(p.get_value() * self.numpy_floatX(0.),
                                     name='%s_rup2' % k)
                       for k, p in tparams.iteritems()]
        running_grads2 = [theano.shared(p.get_value() * self.numpy_floatX(0.),
                                        name='%s_rgrad2' % k)
                          for k, p in tparams.iteritems()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]

        f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                        name='adadelta_f_grad_shared')

        updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
                 for zg, ru2, rg2 in zip(zipped_grads,
                                         running_up2,
                                         running_grads2)]
        ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
                 for ru2, ud in zip(running_up2, updir)]
        param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

        f_update = theano.function([lr], [], updates=ru2up + param_up,
                                   on_unused_input='ignore',
                                   name='adadelta_f_update')

        return f_grad_shared, f_update

    # @classmethod
    def _build_model(self, tparams, options):

        trng = RandomStreams(1234)

        # Used for dropout.
        use_noise = theano.shared(self.numpy_floatX(0.))

        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype=config.floatX)
        y = tensor.vector('y', dtype='int64')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                    n_samples,
                                                    options['dim_proj']])
        proj = self.get_layer(options['encoder'])[1](tparams, emb, options,
                                                prefix=options['encoder'],
                                                mask=mask)
        ### Average the results of the layer----------------------------------------
        if options['encoder'] == 'lstm':
            proj = (proj * mask[:, :, None]).sum(axis=0)
            proj = proj / mask.sum(axis=0)[:, None]
        if options['use_dropout']:
            proj = self.dropout_layer(proj, use_noise, trng)

        pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U'])+tparams['b'])

        f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
        f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

        cost = -tensor.log(pred[tensor.arange(n_samples), y] + 1e-8).mean()

        return use_noise, x, mask, y, f_pred_prob, f_pred, cost

    # @classmethod
    def build_model(self):

        print 'Building model'

        try:
            self.update_options()
            self._params  = self._init_params(self.model_options)
            self._tparams = self._init_tparams(self._params)
            self.optimizer = self.model_options['optimizer']
        except:
            print "Need to preprocess the data first"
            self.preprocess()
            self.update_options()
            self._params  = self._init_params(self.model_options)
            self._tparams = self._init_tparams(self._params)
            self.optimizer = self.model_options['optimizer']

        if self.optimizer=='adadelta':

            optimizer = self.adadelta

            # use_noise is for dropout
            (use_noise, x, mask, y, self.f_pred_prob, self.f_pred, cost) = self._build_model(self._tparams, self.model_options)

            print 'Done. Setting up Optimization Function'

            decay_c = self.model_options['decay_c']
            if decay_c > 0.:
                decay_c = theano.shared(self.numpy_floatX(decay_c), name='decay_c')
                weight_decay = 0.
                weight_decay += (self._tparams['U'] ** 2).sum()
                weight_decay *= decay_c
                cost += weight_decay
                print "cost updated with a weight decay"

            grads = tensor.grad(cost, wrt=self._tparams.values())
            print "gradients set up"

            lr = tensor.scalar(name='lr')
            self.f_grad_shared, self.f_update = optimizer(lr, self._tparams, grads, x, mask, y, cost)
            print "Optimization functions created"

            self.f_cost = theano.function([x, mask, y], cost, name='f_cost')
            print "Cost function created"
            self.f_grad = theano.function([x, mask, y], grads, name='f_grad')
            print "Gradient function created"
        else:
            print "We do not have any other optimizers, stop being difficult and choose one we already have ready."

        # Save variables for us in training section
        self._use_noise = use_noise
        print 'Model Ready!'
        return self

#=======================================================================================================================
# TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING TRAINING
#=======================================================================================================================

    @staticmethod
    def get_minibatches_idx(n, minibatch_size, shuffle=False):
        """
        Used to shuffle the dataset at each iteration.
        """

        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                        minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)

    @staticmethod
    def _prepare_data(seqs, labels, maxlen=None):	# maxlen means how many
        """Create the matrices from the datasets.

        This pad each sequence to the same length: the length of the
        longuest sequence or maxlen.

        if maxlen is set, we will cut all sequence to this maximum
        lenght.

        """
        # x: a list of sentences
        lengths = [len(s) for s in seqs]	### A list contains each of the comments length

        if maxlen is not None:
            new_seqs = []
            new_labels = []
            new_lengths = []
            for l, s, y in zip(lengths, seqs, labels):
                if l < maxlen:
                    new_seqs.append(s)
                    new_labels.append(y)
                    new_lengths.append(l)
            lengths = new_lengths
            labels = new_labels
            seqs = new_seqs			### Filter out the comments whose length exceeds the maxlen

            if len(lengths) < 1:
                return None, None, None

        n_samples = len(seqs)
        maxlen = np.max(lengths)

        x = np.zeros((maxlen, n_samples)).astype('int64')
        x_mask = np.zeros((maxlen, n_samples)).astype('float32')
        for idx, s in enumerate(seqs):
            x[:lengths[idx], idx] = s
            x_mask[:lengths[idx], idx] = 1.

        return x, x_mask, labels	### x is the matrix with dimension of maxlen * n_samples

    @staticmethod
    def unzip(zipped):
        """
        When we pickle the model. Needed for the GPU stuff.
        """
        new_params = OrderedDict()
        for kk, vv in zipped.iteritems():
            new_params[kk] = vv.get_value()
        return new_params

    #______________________________________________________
    # CLASSMETHODS CLASSMETHODS CLASSMETHODS CLASSMETHODS

    # @classmethod
    # def _classify_one(self, data, valid_index):
    #     x, mask, y = self._prepare_data([data[0][t] for t in valid_index],
    #                                      np.array(data[1])[valid_index],
    #                                      maxlen=None)
    #     preds = self.f_pred(x, mask)
    #     return preds

    # @classmethod
    def classify(self, sentences):
        """This function uses f_pred to classify the user provided text.
        To accomplish this, the vector must be reorganized relative to the input DICTIONARY
        sentences is a list of strings
        """
        import time

        start = time.time()

        # empty sentences are an issue
        # xx,sentences = [s,sent for s,sent in enumerate(dirty_sentences): if sent!='']

        data = self._format_sentence_frequencies(sentences)
        # preds = []
        len_data = len(data)
        # iterator = self.get_minibatches_idx(len_data, self.model_options['batch_size'], shuffle=True)
        # for _, valid_index in iterator:
        # for vix in xrange(len_data):
            # this_pred = self._classify_one(data, valid_index)
            # x, mask, y = self._prepare_data([data[t] for t in valid_index], np.array([1]*len_data), maxlen=None)
        x, mask, y = self._prepare_data(data, np.array([1]*len_data), maxlen=None)
        preds = self.f_pred(x, mask)
            # preds.append(this_pred.sum() / len(valid_index))
            # preds.append(this_pred)

        end = time.time()
        print "Time to Tag: " + str(end - start)

        return preds

    # @classmethod
    def pred_error(self, data, iterator, verbose=False):
        """
        Compute the error of the prediction and the truth of the training data.

        :param data: The sentences and their class in a tuple
        :type data: list of 1x2 tuples
        :param_iterator: IDFK
        :rtype:
        """
        valid_err = 0
        for _, valid_index in iterator:
            x, mask, y = self._prepare_data([data[0][t] for t in valid_index],
                                      np.array(data[1])[valid_index],
                                      maxlen=None)
            preds = self.f_pred(x, mask)
            # preds = self._classify_one(data, valid_index)
            targets = np.array(data[1])[valid_index]
            valid_err += (preds == targets).sum()
        valid_err = 1. - self.numpy_floatX(valid_err) / len(data[0])

        return valid_err

    # @classmethod
    def zipp(self, params):
        """
        When we reload the model. Needed for the GPU stuff.
        """
        for kk, vv in params.iteritems():
            self._tparams[kk].set_value(vv)

    # @classmethod
    def train_model(self, max_epochs=None):

        if max_epochs==None:
            max_epochs = self.model_options['max_epochs']
        else:
            self.model_options['max_epochs'] = max_epochs

        kf_valid = self.get_minibatches_idx( len(self.valid_set[0]), self.model_options['valid_batch_size'])
        kf_test  = self.get_minibatches_idx( len(self.test_set[0]) , self.model_options['valid_batch_size'])

        print "%d train examples" % len(self.train_set[0])
        print "%d valid examples" % len(self.valid_set[0])
        print "%d test examples" % len(self.test_set[0])
        history_errs = []
        best_p = None
        bad_count = 0

        if self.model_options['validFreq'] == -1:
            self.model_options['validFreq'] = len(self.train_set[0]) / self.model_options['batch_size']
        if self.model_options['saveFreq'] == -1:
            self.model_options['saveFreq'] = len(self.train_set[0]) / self.model_options['batch_size']

        uidx = 0  # the number of update done
        estop = False  # early stop
        start_time = time.clock()
        try:
            for eidx in xrange(max_epochs):
                n_samples = 0

                # Get new shuffled index for the training set.
                kf = self.get_minibatches_idx(len(self.train_set[0]), self.model_options['batch_size'], shuffle=True)

                for _, train_index in kf:
                    uidx += 1
                    self._use_noise.set_value(1.)

                    # Select the random examples for this minibatch
                    y = [self.train_set[1][t] for t in train_index]
                    x = [self.train_set[0][t]for t in train_index]

                    # Get the data in numpy.ndarray format
                    # This swap the axis!
                    # Return something of shape (minibatch maxlen, n samples)
                    x, mask, y = self._prepare_data(x, y)
                    n_samples += x.shape[1]

                    cost = self.f_grad_shared(x, mask, y)
                    self.f_update(self.model_options['lrate'])

                    if np.isnan(cost) or np.isinf(cost):
                        print 'NaN detected'
                        return 1., 1., 1.

                    if np.mod(uidx, self.model_options['dispFreq']) == 0:
                        print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                    if self.model_options['saveto'] and np.mod(uidx, self.model_options['saveFreq']) == 0:
                        print 'Saving...',

                        if best_p is not None:
                            params = best_p
                        else:
                            params = self.unzip(self._tparams)
                        np.savez(self.model_options['saveto'], history_errs=history_errs, **params)
                        pkl.dump(self.model_options, open('%s.pkl' % self.model_options['saveto'], 'wb'), -1)
                        print 'Done'

                    if np.mod(uidx, self.model_options['validFreq']) == 0:
                        self._use_noise.set_value(0.)
                        train_err = self.pred_error( self.train_set, kf)
                        valid_err = self.pred_error( self.valid_set, kf_valid)
                        test_err  = self.pred_error( self.test_set,  kf_test)

                        history_errs.append([valid_err, test_err])

                        if (uidx == 0 or
                            valid_err <= np.array(history_errs)[:,0].min()):

                            best_p = self.unzip(self._tparams)
                            bad_counter = 0

                        print ('Train ', train_err,
                               'Valid ', valid_err,
                               'Test ', test_err)

                        if (len(history_errs) > self.model_options['patience'] and
                            valid_err >= np.array(history_errs)[:-self.model_options['patience'],0].min()):
                            bad_counter += 1
                            if bad_counter > self.model_options['patience']:
                                print 'Early Stop!'
                                estop = True
                                break

                print 'Seen %d samples' % n_samples

                if estop:
                    break

        except KeyboardInterrupt:
            print "Training interupted"

        end_time = time.clock()
        if best_p is not None:
            self.zipp(best_p)
        else:
            best_p = self.unzip(self._tparams)

        self._use_noise.set_value(0.)
        kf_train_sorted = self.get_minibatches_idx(len(self.train_set[0]), self.model_options['batch_size'])
        train_err = self.pred_error( self.train_set, kf_train_sorted)
        valid_err = self.pred_error( self.valid_set, kf_valid)
        test_err  = self.pred_error( self.test_set,  kf_test)

        print ('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)

        if self.model_options['saveto']:
            np.savez(self.model_options['saveto'], train_err=train_err,
                        valid_err=valid_err, test_err=test_err,
                        history_errs=history_errs, **best_p)
        print 'The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
        print >> sys.stderr, ('Training took %.1fs' %
                              (end_time - start_time))

        # return train_err, valid_err, test_err
        self.train_err = train_err
        self.valid_err = valid_err
        self.test_err = test_err
        return self

    # @classmethod
    def test_model(self):
        return self

    # @classmethod
    def pickle_model(self, filename="./IdeaNet.pkl"):
        '''Save the IdeaNet for future use
            filename: string giving the path and name of the file to save the IdeaNet to
        '''
        id = str(uuid.uuid1())
        if 'pkl' not in filename:
            filename += '.pkl'
        filename += '.' + id
        f = file(filename,'wb')
        state = self.__getstate__()
        pkl.dump(state,f,protocol=pkl.HIGHEST_PROTOCOL)
        f.close()

    # @classmethod
    def load_pickle(self,filename):
        f = file(filename,'rb')
        state = pkl.load(f)
        f.close()
        self.__setstate__(state)
        # self.__init__(IdeaNet, orig='New')

    # @classmethod
    def zip_model(self, filename="./IdeaNet.zip"):
        id = str(uuid.uuid1())
        filename += '.' + id
        f = open(filename,'w')
        theano.misc.pkl_utils.dump(self,f)

    # @classmethod
    def load_zip(self,filename):
        IdeaNet = np.load(filename)
        self.__init__(IdeaNet['self'],orig='New')

    def __getstate__(self,save_training_data=False):

        # Reduce model size by saving off the training data.
        if save_training_data:
            id = str(uuid.uuid1())
            filename = "TrainingSet." + id + ".pickle"
            f = file(filename,'wb')
            pkl.dump(self.train_set,f,protocol=pkl.HIGHEST_PROTOCOL)
            f.close()
        # Return the state and delete the saved training data.
        # ['f_pred', '_use_noise', '_text_col', 'f_cost', '_tparams', 'model_options', 'train_set', '_train_xx', '_layers', 'f_grad_shared', '_test_xx', 'optimizer', 'valid_set', '_data_file', 'f_pred_prob', '_test_size', '_train_size', 'f_update', '_params', 'test_set', 'f_grad', '_n_words', '_label_col', '_data_directory', '_DICTIONARY', '_model_options']
        # Can't save class instances or shared variables due to recursion limits
        state = dict(self.__dict__)
        for key in self._del_keys:
            del state[key]
        return state

    def __setstate__(self, d, train_set_file=None):

        self.__dict__.update(d)

        if train_set_file!=None:
            f = file(train_set_file,'rb')
            d.train_set = pkl.load(f)
            f.close

# if __name__ == '__main__':
#     filename = sys.argv[0]
#     build_model() # Can't actually call it this way right now...