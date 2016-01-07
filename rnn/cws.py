import numpy as np
import theano
import theano.tensor as T
from rnn import RNN
from utils import *
import time
import sys
import os

class CWS:
    def __init__(self, s):
	self.rnn = RNN(s['ne'], s['de'], s['win'], s['nh'], s['nc'], np.random.RandomState(s['seed']))
	self.s = s

    def fit(self, lex, label):
	s = self.s
	n_sentences = len(lex)
	n_train = int(n_sentences * (1. - s['valid_size']))
	s['clr'] = s['lr']
	best_f = 0
	be = 0
	for e in xrange(s['n_epochs']):
	    shuffle([lex, label], s['seed'])
	    train_lex, valid_lex = lex[:n_train], lex[n_train:]
	    train_label, valid_label = label[:n_train], label[n_train:]
	    tic = time.time()
	    for i in xrange(n_train):
		cwords = contextwin(train_lex[i], s['win'])
		words = map(lambda x: np.asarray(x).astype('int32'), minibatch(cwords, s['bs']))
		labels = train_label[i]
		for word_batch, label_last_word in zip(words, labels):
		    self.rnn.fit(word_batch, label_last_word, s['clr'])
		    self.rnn.normalize()
		    if s['verbose']:
			print '[learning] epoch %i >> %2.2f%%' % (e+1, (i+1)*100./n_train), 'completed in %s << \r' % time_format(time.time() - tic),
			sys.stdout.flush()

	    pred_y = self.predict(valid_lex)
	    p, r, f = evaluate(pred_y, valid_label)
	    print '[learning] epoch %i >> P: %2.2f%% R: %2.2f%% F: %2.2f%%' % (e+1, p*100., r*100., f*100.), '<< %s used' % time_format(time.time() - tic)
	    
	    if f > best_f:
		best_f = f
		be = e
		self.save()
    
	    if s['decay'] and e - be >= 5: s['clr'] *= 0.5	    
	    if s['clr'] < 1e-5: break

    def predict(self, lex):
	s = self.s
	y = [self.rnn.predict(np.asarray(contextwin(x, s['win'])).astype('int32'))[1:-1] for x in lex]
	return y

    def save(self):
	if not os.path.exists('params'): os.mkdir('params')
	self.rnn.save() 

    def load(self):
	self.rnn.load()

