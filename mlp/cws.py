import numpy as np
import theano
import theano.tensor as T
from mlp import MLP
from utils import *
import time
import sys
import os

class CWS:
    def __init__(self, s):
	self.mlp = MLP(s['ne'], s['de'], s['win'], s['nh'], 4, s['L2_reg'], np.random.RandomState(s['seed']))
	self.s = s

    def fit(self, lex, label):
	s = self.s
	n_sentences = len(lex)
	n_train = int(n_sentences * (1. - s['valid_size']))
	s['clr'] = s['lr']
	best_f = 0
	for e in xrange(s['n_epochs']):
	    shuffle([lex, label], s['seed'])
	    train_lex, valid_lex = lex[:n_train], lex[n_train:]
	    train_label, valid_label = label[:n_train], label[n_train:]
	    tic = time.time()
	    cost = 0
	    for i in xrange(n_train):
		if len(train_lex[i]) == 2: continue
		words = np.asarray(contextwin(train_lex[i], s['win']), dtype = 'int32')
		labels = [0] + train_label[i] + [0]
		y_pred = self.mlp.predict(words)
		cost += self.mlp.fit(words, [0]+y_pred, [0]+labels, s['clr'])
		self.mlp.normalize()
		if s['verbose']:
		    print '[learning] epoch %i >> %2.2f%%' % (e+1, (i+1)*100./n_train), 'completed in %s << \r' % time_format(time.time() - tic),
		    sys.stdout.flush()
	    print '[learning] epoch %i >> cost = %f' % (e+1, cost / n_train), ', %s used' % time_format(time.time() - tic)
	    pred_y = self.predict(valid_lex)
	    p, r, f = evaluate(pred_y, valid_label)
	    print '           P: %2.2f%% R: %2.2f%% F: %2.2f%%' % (p*100., r*100., f*100.)
	    '''
	    if f > best_f:
		best_f = f
		self.save()
	    '''

    def predict(self, lex):
	s = self.s
	y = [self.mlp.predict(np.asarray(contextwin(x, s['win'])).astype('int32'))[1:-1] for x in lex]
	return y

    def save(self):
	if not os.path.exists('params'): os.mkdir('params')
	self.mlp.save() 

    def load(self):
	self.mlp.load()

