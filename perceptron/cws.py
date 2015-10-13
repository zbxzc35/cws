# encoding: utf-8

from vector import *
from beam import *
import weight
import os
import time
import re

punctuation = u'、，。！？：；（）《》“”'

class CWS:

    def tick(self):
	self.time = time.time()

    def tock(self):
	t = time.time() - self.time
	self.tick()
	return t

    def train(self, train_samples, P = 1):
	print '-'*50
	print 'Training started...'
	self.tick()
	w = Vector()
	N = len(train_samples)
	print '%d training samples found.' % N
	count = 0
	for times in xrange(0, P):
	    for sample in train_samples:
		x = sample[0]
		y = sample[1]
		Z = decode(x)
		if y != Z.get_tags():
		    vy = State(x, y).get_feature_vector()
		    vz = Z.get_feature_vector()
		    weight.W.add_batch(w, vy, count)
		    weight.W.add_batch(w, vz, count)
		    w.add(vy)
		    w.minus(vz)
		    weight.W.add(vy)
		    weight.W.minus(vz)
		count += 1
		if count % 100 == 0: print count
	weight.W.add_batch(w, w, count)
	weight.W.divide(count)
	print 'Training finished.'
	time_used = self.tock()
	print 'Time used: %.2f s' % time_used
	print 'Time used / 100 Sentence: %.2f s' % (time_used * 100 / count)

    def save(self, path = 'dat/'):
	print '-'*50
	print 'Saving...'
	self.tick()
	if len(path) > 0 and path[-1] != '/': path.append('/')
	if not os.path.exists(path): os.mkdir(path)
	for i in xrange(0, 14):
	    with open('dat/w%d.dat' % (i+1), 'w') as fout:
		for feature in weight.W.v[i]:
		    if weight.W.v[i][feature] == 0: continue
		    fout.write('%s\t%f\n' % (feature.encode('utf-8'), weight.W.v[i][feature]))
	print 'Saved.'
	time_used = self.tock()
	print 'Time used: %.2f s' % time_used

    def load(self, path = 'dat/'):
	print '-'*50
	print 'Loading...'
	self.tick()
	if len(path) > 0 and path[-1] != '/': path.append('/')
	if not os.path.exists(path): return
	for i in xrange(0, 14):
	    with open(path + 'w%d.dat' % (i+1), 'r') as fin:
		lines = fin.readlines()
		for line in lines:
		    (feature, w) = line.rstrip().decode('utf8').split('\t')
		    w = float(w)
		    weight.W.v[i][feature] = w
	print 'Loaded.'
	time_used = self.tock()
	print 'Time used: %.2f s' % time_used

    def predict(self, sentence):
	if sentence == '': return ''
	m = re.split(u'([%s])' % punctuation, sentence)
	seg = []
	for s in m:
	    if s in punctuation: seg.append(s)
	    else: seg += self.predict_one(s)
	return seg

    def predict_one(self, sentence):
	if sentence == '': return ''
	best = decode(sentence)
	return best.get_segment()

