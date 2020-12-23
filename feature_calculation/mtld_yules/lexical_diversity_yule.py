import collections
import re
import sys

def get_yules(s):
    """ 
    Returns a tuple with Yule's K and Yule's I.
    (cf. Oakes, M.P. 1998. Statistics for Corpus Linguistics.
    International Journal of Applied Linguistics, Vol 10 Issue 2)

    In production this needs exception handling.
    """
    tokens = s
# AT do not uppercase
#    token_counter = collections.Counter(tok.upper() for tok in tokens)
    token_counter = collections.Counter(tok.lower() for tok in tokens)
#    token_counter = collections.Counter(tok for tok in tokens)

    m1 = sum(list(token_counter.values()))
    m2 = sum([freq ** 2 for freq in token_counter.values()])
    i = (m1*m1) / (m2-m1)
    k = 1/i * 10000
    print(i)#,i)
    #return (k, i)
    return(i)


with open(sys.argv[1]) as f:
	data = f.read().split()
	get_yules(data)

