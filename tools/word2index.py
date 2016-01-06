import collections
import pickle

def build_dictionary(words, vocabulary_size = 50000):
    count = [('<UNK>', -1), ('<BOS>', -1), ('<EOS>', -1)]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
	dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def word2index(words, dictionary):
    return map(lambda w: dictionary.get(w, 0), words)

def seg2tag(segs):
    tags = []
    for seg in segs:
	n = len(seg)
	if n == 1:
	   tags += [0]
	else:
	   tags += [1] + [3] * (n-2) + [2]
    return tags

if __name__ == '__main__':
    with open('../training/pku.utf8', 'r') as fin:
	lines = fin.readlines()
	lines = [line.strip().decode('utf-8') for line in lines]
	
	words = ''.join([''.join(line.split('  ')) for line in lines])
	dic, rdic = build_dictionary(words)
	
	with open('../training/pku_dic.pkl', 'wb') as fout:
	    pickle.dump(dic, fout)
	with open('../training/pku_rdic.pkl', 'wb') as fout:
	    pickle.dump(rdic, fout)

	x_train = []
	y_train = []
	for line in lines:
	    segs = line.split('  ')
	    idxs = word2index(['<BOS>'] + list(''.join(segs)) + ['<EOS>'], dic)
	    tags = seg2tag(segs)
	    x_train.append(idxs)
	    y_train.append(tags)
	
	with open('../training/pku_lex.pkl', 'wb') as fout:
	    pickle.dump(x_train, fout)
	with open('../training/pku_label.pkl', 'wb') as fout:
	    pickle.dump(y_train, fout)

    with open('../testing/pku_gold.utf8', 'r') as fin:
	lines = fin.readlines()
	lines = [line.strip().decode('utf-8') for line in lines]

	x_test = []
	y_test = []
	for line in lines:
	    segs = line.split('  ')
	    idxs = word2index(['<BOS>'] + list(''.join(segs)) + ['<EOS>'], dic)
	    tags = seg2tag(segs)
	    x_test.append(idxs)
	    y_test.append(tags)
	
	with open('../testing/pku_lex.pkl', 'wb') as fout:
	    pickle.dump(x_test, fout)
	with open('../testing/pku_label.pkl', 'wb') as fout:
	    pickle.dump(y_test, fout)
	
