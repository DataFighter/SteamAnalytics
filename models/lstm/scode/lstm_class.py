#!/usr/bin/env python

import theano, copy
import theano.tensor as tensor
from theano import config
import numpy as np
from collections import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
# from lstm_utils import *


class LSTM(object):

    @classmethod
    def __init__(self,PD): # PD = Params Data

        self.PD = PD
        self._layers = {'lstm': (self.param_init_lstm, self.lstm_layer)}
        self.model_options = PD.model_options
        self._params  = self._init_params(self.model_options)
        self._tparams = self._init_tparams(self._params)
        self.model_options = PD.model_options
        self.optimizer = self.model_options['optimizer']

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

    @classmethod
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

    @classmethod
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

    @classmethod
    def get_layer(self, name):
        fns = self._layers[name]
        return fns

    @staticmethod
    def dropout_layer( state_before, use_noise, trng):
        proj = tensor.switch(use_noise,
                             (state_before *
                              trng.binomial(state_before.shape,
                                            p=0.5, n=1,
                                            dtype=state_before.dtype)),
                             state_before * 0.5)
        return proj

    @classmethod
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

    @classmethod
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

    @classmethod
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

    @classmethod
    def build_model(self):

        print 'Building model'

        if self.optimizer=='adadelta':

            optimizer = self.adadelta

            # use_noise is for dropout
            (use_noise, x, mask, y, f_pred_prob, f_pred, cost) = self._build_model(self._tparams, self.model_options)

            print 'Done. Setting up Optimization Function'

            decay_c = self.model_options['decay_c']
            if decay_c > 0.:
                decay_c = theano.shared(self.numpy_floatX(decay_c), name='decay_c')
                weight_decay = 0.
                weight_decay += (self._tparams['U'] ** 2).sum()
                weight_decay *= decay_c
                cost += weight_decay

            grads = tensor.grad(cost, wrt=self._tparams.values())

            lr = tensor.scalar(name='lr')
            self.f_grad_shared, self.f_update = optimizer(lr, self._tparams, grads, x, mask, y, cost)

            f_cost = theano.function([x, mask, y], cost, name='f_cost')
            f_grad = theano.function([x, mask, y], grads, name='f_grad')
        else:
            print "We do not have any other optimizers, stop being difficult and choose one we already have ready."

        print 'Model Ready!'
        return self

    @classmethod
    def train_model(self):
        return self

    @classmethod
    def test_model(self):
        return self

# if __name__ == '__main__':
#     filename = sys.argv[0]
#     build_model() # Can't actually call it this way right now...