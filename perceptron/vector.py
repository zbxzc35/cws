# encoding: utf-8

import feature_template

last = []
for i in xrange(feature_template.feature_num):
    last.append({})

class Vector:
    feature_num = feature_template.feature_num

    def __init__(self):
	self.v = []
	for i in xrange(self.feature_num):
	    self.v.append({})

    def get(self, i, feature):
	return self.v[i].get(feature, 0.0)

    def add(self, target):
	for i in xrange(self.feature_num):
	    for feature in target.v[i]:
		self.v[i][feature] = self.v[i].get(feature, 0.0) + target.v[i][feature]

    def minus(self, target):
	for i in xrange(self.feature_num):
	    for feature in target.v[i]:
		self.v[i][feature] = self.v[i].get(feature, 0.0) - target.v[i][feature]

    def add_one(self, i, feature):
    	self.v[i][feature] = self.v[i].get(feature, 0.0) + 1.0

    def divide(self, k):
	for i in xrange(self.feature_num):
	    for feature in self.v[i]:
		self.v[i][feature] /= k
    
    def add_batch(self, target, features, count):
	for i in xrange(self.feature_num):
	    for feature in features.v[i]:
		last_count = last[i].get(feature, 0)
		self.v[i][feature] = self.v[i].get(feature, 0.0) + target.v[i].get(feature, 0) * (count - last_count)
		last[i][feature] = count
