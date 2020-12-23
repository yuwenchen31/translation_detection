#!/usr/bin/env python3

# MTLD and HD-D are measures for lexical diversity that do not suffer the drawbacks of TTR

import sys
sys.path.insert(1, '../third/')
from lexical_diversity import mtld, hdd

with open(sys.argv[1]) as f:
	data = f.read().split()

#print(mtld(data))
print(hdd(data))



