#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 17:50:13 2020

@author: yuwen
"""

#to get 40m training sentences for training langauge models 

import random
import bz2
import sys

data = []

print(sys.argv)

obj = sys.argv[1]
out = sys.argv[2]


with open(obj, 'rb') as f:
    with bz2.open(out,'wb') as outf:
        data = f.read().splitlines()
        
        shorten_data = random.sample(data,40000000)  
        
        for line in shorten_data: 
            #print(line)
            #print(type(line))
            outf.write(line + b'\n')
            #print(line, file=outf)
            
        outf.close()
