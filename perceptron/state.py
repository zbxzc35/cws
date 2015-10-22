# encoding: utf-8

import weight
from vector import *
from feature_template import *

class State:
    tags = 'BMES'

    def __init__(self, S, T = '', k = 0, n = 0, score = 0):
	self.S = S
	if n == 0: self.n = len(S)
	else: self.n = n
	self.T = T
	self.k = k
	self.score = score
	if k: self.cal_score()

    def get_tags(self):
	return self.T

    def get_segment(self):
    	seg = []
	for c, t in zip(self.S, self.T):
	    if t == 'B' or t == 'S': seg.append(c)
	    else: seg[-1] += c
	return seg

    def get_score(self):
	return self.score

    def expand(self):
	tags = self.tags
	if not self.T: tags = 'BS'
	else:
	    if self.T[-1] == 'B' or self.T[-1] == 'M': tags = 'ME'
	    else: tags = 'BS'
	if self.k + 1 == self.n:
	    tmp = ''
	    for t in tags:
		if t in 'ES':
		    tmp += t
	    tags = tmp
	for tag in tags:
	    yield State(self.S, self.T + tag, self.k+1, self.n, self.score)

    def get_feature_vector(self):
	feature_vector = Vector()
	for i in xrange(self.n):
	    features = get_features(self.S, self.T, i, self.n)
	    for feature in features:
		feature_vector.add_one(feature[0], feature[1])
	return feature_vector

    def cal_score(self):
	score = 0
	features = get_features(self.S, self.T, self.k-1, self.n)
	for feature in features:
	    score += weight.W.get(feature[0], feature[1])
	self.score += score
		
