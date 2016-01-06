# encoding: utf-8

from cws import *
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) > 1:    
	output_path = sys.argv[1]
    else:
	output_path = '/home/huangshenno1/cws/result.utf8'

    test_path = '/home/huangshenno1/cws/testing/CTB.gb'
    with open(test_path, 'r') as test_file:
	lines = test_file.readlines()
    sentences = [''.join(line.strip().decode('gb2312').split()) for line in lines]

    cws = CWS()	
    cws.load()

    with open(output_path, 'w') as fout:
	num = 0
        for sentence in sentences:
	    num += 1
	    seg = [x.encode('utf-8') for x in cws.predict(sentence)]
	    # print u'/'.join(seg)
	    fout.write(' '.join(seg)+'\n')
	    if num % 100 == 0: print num
