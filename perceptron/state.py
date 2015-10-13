# encoding: utf-8

import weight
from vector import *
from feature_template import *

class State:
    tags = 'BI'

    def __init__(self, S, T = '', n = 0, score = 0):
	self.S = S
	if n == 0: self.n = len(S)
	else: self.n = n
	self.T = T
	self.score = score
	if T: self.cal_score()

    def get_tags(self):
	return self.T

    def get_segment(self):
    	seg = []
	for c, t in zip(self.S, self.T):
	    if t == 'B': seg.append(c)
	    else: seg[-1] += c
	return seg

    def get_score(self):
	return self.score

    def expand(self):
	tags = self.tags
	if not self.T: tags = 'B'
	for tag in tags:
	    yield State(self.S, self.T + tag, self.n, self.score)

    def get_feature_vector(self):
	feature_vector = Vector()
	for i in xrange(self.n):
	    features = get_features(self.S, self.T, i)
	    for feature in features:
		feature_vector.add_one(feature[0], feature[1])
	return feature_vector

    def cal_score(self):
	score = 0
	features = get_features(self.S, self.T, len(self.T)-1)
	for feature in features:
	    score += weight.W.get(feature[0], feature[1])
	self.score += score
		
