# encoding: utf-8

import heapq
from state import *

def decode(sentence):
    return beam_search(State(sentence), len(sentence), 16)

def beam_search(start_item, n, B):
    candidates = [start_item]
    for i in xrange(n):
	agenda = []
	for candidate in candidates:
    	    agenda += list(candidate.expand())
	if i == n-1:
	    best = top(agenda)
	    return best
	candidates = topB(agenda, B)

def top(agenda):
    return max(agenda, key = lambda x: x.score)

def topB(agenda, B):
    return heapq.nlargest(B, agenda, key = lambda x: x.score)
	
