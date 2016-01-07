from cws import CWS
from utils import *
import pickle

if __name__ == '__main__':
    s = {'de': 50, 
	'nh': 100,
	'win': 5,
	'lr': 0.05,
	'L2_reg': 0.0001,
	'valid_size': 0.2,
	'n_epochs': 100,
	'seed': 1231,
	'verbose': 1}
    with open('hyperparam.pkl', 'wb') as fout:
	pickle.dump(s, fout)

    with open('../training/pku_lex.pkl', 'rb') as fin:
	train_lex = pickle.load(fin)
    with open('../training/pku_label.pkl', 'rb') as fin:
	train_label = pickle.load(fin)
    with open('../testing/pku_lex.pkl', 'rb') as fin:
	test_lex = pickle.load(fin)
    with open('../testing/pku_label.pkl', 'rb') as fin:
	test_label = pickle.load(fin)
    with open('../training/pku_rdic.pkl', 'rb') as fin:
	rdic = pickle.load(fin)
    s['ne'] = len(rdic)

    N = 500
    train_lex = train_lex[:N]
    train_label = train_label[:N]
    test_lex = test_lex[:N]

    cws = CWS(s)
    cws.fit(train_lex, train_label)
    y = cws.predict(test_lex)
    p, r, f = evaluate(y, test_label)
    print '='*39
    print '[testing] P: %2.2f%% R: %2.2f%% F: %2.2f%%' % (p*100., r*100., f*100.)
    
    with open('ans.utf8', 'w') as fout:
	for sent, tags in zip(test_lex, y):
	    fr = True
	    for c, t in zip(sent[1:-1], tags):
		if fr: fr = False
		elif t == 0 or t == 1: fout.write(' ')
		fout.write(rdic[c].encode('utf-8'))
	    fout.write('\n')
