import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict

class Layer:
    def __init__(self, n_in, n_out, W = None, b = None, rng = np.random.RandomState()):
	if W is None:
	    W_values = np.asarray(rng.uniform(
		low = -np.sqrt(6. / (n_in + n_out)),
		high = np.sqrt(6. / (n_in + n_out)),
		size = (n_in, n_out)), dtype = theano.config.floatX)
	    W_values *= 4
	    W = theano.shared(value = W_values, name = 'W', borrow = True)

	if b is None:
	    b_values = np.zeros((n_out,), dtype = theano.config.floatX)
	    b = theano.shared(value = b_values, name = 'b', borrow = True)

	self.W = W
	self.b = b

	self.params = [self.W, self.b]

class MLP:
    def __init__(self, ne, de, cs, nh, nc, L2_reg = 0.0, rng = np.random.RandomState()):
	self.nc = nc
	self.hiddenLayer = Layer(de*cs, nh, rng = rng)
	self.outputLayer = Layer(nh, nc)
	self.emb = theano.shared(rng.normal(loc = 0.0, scale = 0.01, size = (ne, de)).astype(theano.config.floatX))
	A = rng.normal(loc = 0.0, scale = 0.01, size = (nc, nc)).astype(theano.config.floatX)
	self.A = theano.shared(value = A, name = 'A', borrow = True)

	self.params = self.hiddenLayer.params + self.outputLayer.params + [self.emb, self.A]
	self.names = ['Wh', 'bh', 'w', 'b', 'emb', 'A']

	idxs = T.imatrix('idxs')
	x = self.emb[idxs].reshape((idxs.shape[0], de*cs))
	y = T.bvector('y')
	ans = T.bvector('ans')

	INF = 1e9
	result, updates1 = theano.scan(fn = self.one_step, sequences = x, outputs_info = [theano.shared(0.0), theano.shared(-INF), theano.shared(-INF), theano.shared(-INF), None, None, None, None])
	self.decode = theano.function(inputs = [idxs], outputs = result, updates = updates1)

	score, updates2 = theano.scan(fn = self.two_step, sequences = [x, dict(input = y, taps = [-1, 0]), dict(input = ans, taps = [-1, 0])], outputs_info = theano.shared(0.0))

	cost = score[-1]
	gradients = T.grad(cost, self.params)
	lr = T.scalar('lr')
	for p, g in zip(self.params, gradients):
	    updates2[p] = p + lr * g

	self.fit = theano.function(inputs = [idxs, y, ans, lr], outputs = cost, updates = updates2)
	self.normalize = theano.function(inputs = [], updates = {self.emb: self.emb / T.sqrt((self.emb**2).sum(axis=1)).dimshuffle(0, 'x')})

    def one_step(self, x_t, dp0_m1, dp1_m1, dp2_m1, dp3_m1):
	h_t = T.nnet.sigmoid(T.dot(x_t, self.hiddenLayer.W) + self.hiddenLayer.b)
	s_t = T.dot(h_t, self.outputLayer.W) + self.outputLayer.b
	dp0 = [dp0_m1, dp1_m1, dp2_m1, dp3_m1]
	dp = [None] * self.nc
	pre = [-1] * self.nc
	go = [[0,1], [2,3], [0,1], [2,3]]
	for i in xrange(self.nc):
	    if dp0[i] is None: continue
	    for j in go[i]:
		t = dp0[i] + s_t[j] + self.A[i][j]
		if dp[j] is None:
		    dp[j] = t
		    pre[j] = T.constant(i)
		else:
		    m = T.switch(T.gt(t, dp[j]), t, dp[j])
		    pre[j] = T.switch(T.gt(t, dp[j]), T.constant(i), pre[j])
		    dp[j] = m
	return dp + pre
	
    def predict(self, idxs):
	n = len(idxs)
	nc = self.nc
	result = self.decode(idxs[1:])
	pre = np.asarray(result[4:])
	y_pred = [0] * n
	for i in xrange(n-2, 0, -1):
	    y_pred[i] = pre[y_pred[i+1]][i]
	return y_pred

    def two_step(self, x_t, y_tm1, y_t, ans_tm1, ans_t, score_m1):
	h_t = T.nnet.sigmoid(T.dot(x_t, self.hiddenLayer.W) + self.hiddenLayer.b)
	s_t = T.dot(h_t, self.outputLayer.W) + self.outputLayer.b
	score = score_m1 \
	      + (s_t[ans_t] + self.A[ans_tm1][ans_t]) \
	      - (s_t[y_t] + self.A[y_tm1][y_t])
	return score

    def save(self):
	for param, name in zip(self.params, self.names):
	    np.save('params/'+name+'.npy', param.get_value())

    def load(self):
	for param, name in zip(self.params, self.names):
	    param.set_value(np.load('params/'+name+'.npy'))

