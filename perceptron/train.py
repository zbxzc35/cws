# encoding: utf-8

from cws import *
import os

def seg2tag(seg):
    tags = ''
    for s in seg:
	tags += 'B'
	for c in s[1:]:
	   tags += 'I'
    return tags

if __name__ == '__main__':
    train_path = '../training/pku_training.utf8'
    with open(train_path, 'r') as train_file:
	lines = train_file.readlines()
    train_samples = []
    for line in lines:
	line = line.strip().decode('utf-8')
	if line == '': continue
	segments = line.split('  ')
	train_samples.append((''.join(segments), seg2tag(segments)))

    cws = CWS()	
    cws.train(train_samples, 6)
    cws.save()
# cws.load()
    print u'/'.join(cws.predict(u'南京市长江大桥'))
    print u'/'.join(cws.predict(u'中共中央总书记'))
    print u'/'.join(cws.predict(u'部分居民生活水平'))
    print u'/'.join(cws.predict(u'我来到北京清华大学'))
    print u'/'.join(cws.predict(u'小明硕士毕业于中国科学院计算所，后在日本京都大学深造'))
