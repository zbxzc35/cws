# encoding: utf-8

from cws import *
import os

if __name__ == '__main__':
    test_path = '../testing/pku_test.utf8'
    with open(test_path, 'r') as test_file:
	lines = test_file.readlines()
    sentences = [line.strip().decode('utf-8') for line in lines]

    cws = CWS()	
    cws.load()

    with open('../result.utf8', 'w') as fout:
	num = 0
        for sentence in sentences:
	    num += 1
	    seg = [x.encode('utf-8') for x in cws.predict(sentence)]
	    # print u'/'.join(seg)
	    fout.write(' '.join(seg)+'\n')
	    if num % 100 == 0: print num
