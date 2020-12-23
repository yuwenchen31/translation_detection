#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 12:02:07 2020

@author: chenfish
"""

import math
#Kendall's tau distance 

#expect the mgiza++ output as a&b


a = [1,2,3,4,5,6,7,8,9,10]
b = [2,3,4,5,6,7,8,9,10,1]


count = 0 
for i in range(len(a)):
    for j in range(len(a)):
        if j > i:
            if a[i] < a[j] and b[i] > b[j]:
                count +=1
                
                
dk = 1 - math.sqrt( count/( (len(a) **2 - len(a) )/2) )
print(dk)