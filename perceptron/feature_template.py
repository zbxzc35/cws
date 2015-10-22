# encoding: utf-8

feature_num = 23

def get_features(S, T, k, n):
    s = []
    for i in xrange(k+1):
	if T[i] == 'B' or T[i] == 'S': s.append(S[i])
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

    # character-based tagging
    num = 14
    t = T[k]
    # 6
    for i in xrange(k-1, k+2):
	if valid(i, n): 
	    yield (num, S[i]+t)
	    yield (num+1, get_type(S[i])+t)
	num += 2
    # 2
    for i in xrange(k-1, k+1):
	if valid(i, n) and valid(i+1, n): yield (num, S[i]+S[i+1]+t)
	num += 1
    # 1
    if valid(k-1, n) and valid(k+1, n): yield (num, S[k-1]+S[k+1]+t)

def valid(i, n):
    return i >= 0 and i < n

def get_type(c):
    if c >= u'0' and c <= u'9': return '0'
    if c in u'○〇零一二三四五六七八九十百千万亿点': return '0'
    if c in u'年月日': return '1'
    if c >= u'a' and c <= u'z': return '2'
    if c >= u'A' and c <= u'Z': return '2'
    if c in u"．!@#$%^&*()_+=-[]{};:'\"|\\,./<>?~`！@#￥%……&*（）-——+=【】{}、；：‘’“”，。《》？·~！＠＃￥％……＆×（）－——＝＋【】｛｝；：‘’“”，。《》？、／＼℃": return '3'
    return '4'

