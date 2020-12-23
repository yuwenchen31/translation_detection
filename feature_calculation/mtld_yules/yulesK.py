#!/usr/bin/env python

import sys
import string
import math
from collections import defaultdict

def separate_punctuation(line):
    words = line.strip().split()
    tokenized = []
    for w in words:
        if len(w) == 1:
            tokenized.append(w)
        else:
            lastChar = w[-1] 
            firstChar = w[0]
            if lastChar in string.punctuation:
                tokenized += [w[:-1], lastChar]
            elif firstChar in string.punctuation:
                tokenized += [firstChar, w[1:]]
            else:
                tokenized.append(w)
    
    return tokenized

counts = defaultdict(float)
counts_of_counts = defaultdict(float)
total_words = 0.0

for line in sys.stdin:
    #running_words = separate_punctuation(line)
    running_words = line.split()
    total_words += len(running_words)
    for rw in running_words:
        counts[rw]+=1

#print("%s\t%i" % ("total number of words:",total_words))

for c in counts.values():
    counts_of_counts[c]+=1

yulesk = 0.0

for c in counts_of_counts.items():
    yulesk+=c[1]*math.pow(c[0],2)/math.pow(total_words,2)

yulesk = (yulesk - 1.0/total_words)*10000

print("%s\t%f" % ("Yule's K:", yulesk))







