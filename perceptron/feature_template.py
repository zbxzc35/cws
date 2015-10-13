# encoding: utf-8

feature_num = 14

def get_features(S, T, k):
    s = []
    for i in xrange(k+1):
	if T[i] == 'B': s.append(S[i])
	else: s[-1] += S[i]
    c = s.pop()
    if len(c) == 1:
	if len(s) >= 1:
	    yield (0, s[-1])
	    if len(s[-1]) == 1:
		yield (2, s[-1])
	    yield (3, s[-1][0] + str(len(s[-1])))
	    yield (4, s[-1][-1] + str(len(s[-1])))
	    yield (5, s[-1][-1] + c)
	    yield (7, s[-1][0] + s[-1][-1])
	    yield (8, s[-1] + c)
	    yield (10, s[-1][0] + c)
	    if len(s) >= 2:
		yield (1, s[-2] + s[-1])
		yield (9, s[-2][-1] + s[-1])
		yield (11, s[-2][-1] + s[-1][-1])
		yield (12, s[-2] + str(len(s[-1])))
		yield (13, str(len(s[-2])) + s[-1])
    else: 
	yield (6, c[-2:])
