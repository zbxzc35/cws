import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

class RNN:
    def __init__(self, ne, de, cs, nh, nc, rng = np.random.RandomState()):
	self.emb = theano.shared(0.2 * rng.uniform(-1.0, 1.0, (ne+1, de)).astype(theano.config.floatX))
	self.wx = theano.shared(0.2 * rng.uniform(-1.0, 1.0, (de*cs, nh)).astype(theano.config.floatX))
	self.wh = theano.shared(0.2 * rng.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
	self.w = theano.shared(0.2 * rng.uniform(-1.0, 1.0, (nh, nc)).astype(theano.config.floatX))
	self.bh = theano.shared(np.zeros(nh, dtype = theano.config.floatX))
	self.b = theano.shared(np.zeros(nc, dtype = theano.config.floatX))
	self.h0 = theano.shared(np.zeros(nh, dtype = theano.config.floatX))

	self.params = [self.emb, self.wx, self.wh, self.w, self.bh, self.b, self.h0]
	self.names = ['emb', 'wx', 'wh', 'w', 'bh', 'b', 'h0']

	idxs = T.imatrix()
	x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
	y = T.iscalar('y')

	def recurrence(x_t, h_tm1):
	    h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
	    s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
	    return [h_t, s_t]
	
	[h, s], _ = theano.scan(fn = recurrence, sequences = x, outputs_info = [self.h0, None], n_steps = x.shape[0])

	p_y_given_x_lastword = s[-1, 0, :]
	p_y_given_x_sentence = s[:, 0, :]
	y_pred = T.argmax(p_y_given_x_sentence, axis = 1)

	lr = T.scalar('lr')
	nll = -T.log(p_y_given_x_lastword)[y]
	gradients = T.grad(nll, self.params)
	updates = OrderedDict((p, p-lr*g) for p, g in zip(self.params, gradients))

	self.predict = theano.function(inputs = [idxs], outputs = y_pred)
	self.fit = theano.function(inputs = [idxs, y, lr], outputs = nll, updates = updates)
	self.normalize = theano.function(inputs = [], updates = {self.emb: self.emb / T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0, 'x')})

