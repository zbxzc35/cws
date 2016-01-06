from cws import CWS
from utils import *
import pickle

if __name__ == '__main__':
    with open('hyperparam.pkl', 'rb') as fin:
	s = pickle.load(fin)

    with open('../testing/pku_lex.pkl', 'rb') as fin:
	test_lex = pickle.load(fin)
    with open('../testing/pku_label.pkl', 'rb') as fin:
	test_label = pickle.load(fin)
    with open('../training/pku_rdic.pkl', 'rb') as fin:
	rdic = pickle.load(fin)
    s['ne'] = len(rdic)

    cws = CWS(s)
    cws.load()
    y = cws.predict(test_lex)
    p, r, f = evaluate(y, test_label)
    print '[testing] P: %2.2f%% R: %2.2f%% F: %2.2f%%' % (p*100., r*100., f*100.)
    
    with open('ans.utf8', 'w') as fout:
	for sent, tags in zip(test_lex, y):
	    fr = True
	    for c, t in zip(sent[1:-1], tags):
		if fr: fr = False
		elif t == 0 or t == 1: fout.write(' ')
		fout.write(rdic[c].encode('utf-8'))
	    fout.write('\n')
