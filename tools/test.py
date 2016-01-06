import pickle

with open('../training/pku_rdic.pkl', 'rb') as fin:
    dic = pickle.load(fin)
    print dic
