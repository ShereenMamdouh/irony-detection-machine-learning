# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 15:11:46 2017

@author: Sherin
"""

import re
semantic=[]
bagofwords=[]
wordembedding=[]
labels = []

import numpy as np
from random import randint
def readdata(filename):
 mynumber = []
 with open(filename) as f:
    for line in f:
        mynumber.extend([n for n in line.strip().split('_')])
 mynumber = [x.strip(' ') for x in mynumber]
 mynumber = mynumber[:-1]
 mynumber = list(map(float, mynumber))
 return mynumber
nearest=readdata("naive-bag-final.txt")
count=len(nearest)
file = open("predictions-taskA.txt","w") 
nearest= list(map(int, nearest))
print(nearest)
for i in range(0,count):
  file.write("%d " %nearest[i])
  file.write("\n")
file.close()
