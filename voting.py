# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 23:14:23 2017

@author: Sherin
"""
import re
semantic=[]
bagofwords=[]
wordembedding=[]
labels = []
count=100-1
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
def votingsystem(svm,naive,trees,nearest):
    vector=[]
    for i in range(0,count):
     if (svm[i]+naive[i]+trees[i]+nearest[i])>2:
        vector.insert(i,1)
     elif (svm[i]+naive[i]+trees[i]+nearest[i])<2:
        vector.insert(i,0)
     else: 
        vector.insert(i,randint(0, 1))
    return vector
def votingsystem1(bag,semantic,word):
    vector=[]
    for i in range(0,count):
     if (bag[i]+semantic[i]+word[i])>1:
        vector.insert(i,1)
     else:
        vector.insert(i,0)
    return vector
svm =readdata("svm.txt")
naive =readdata( "naive.txt")
trees =readdata( "trees.txt")
nearest=readdata( "nearest.txt")
semantic=votingsystem(svm,naive,trees,nearest)
print('the vote on semantic')
print(semantic)
svm =readdata("svm-emb.txt")
naive=readdata( "naive-emb.txt")
trees =readdata( "trees-emb.txt")
nearest=readdata( "nearest-emb.txt")
bagofwords=votingsystem(svm,naive,trees,nearest)
print('the vote on bag of words')
print(bagofwords)
svm =readdata("svm-bag.txt")
naive1 =readdata( "naive-bag.txt")
trees=readdata( "trees-bag.txt")
nearest=readdata( "nearest-bag.txt")
wordembedding=votingsystem(svm,naive1,trees,nearest)
print('the vote on wordembedding')
print(wordembedding)
totalvotes=votingsystem1(semantic,wordembedding,bagofwords)

with open("SemEval2018-T3-train-taskA.txt", encoding="utf8") as ins:
    array = []
    tweets=[]
    X=[]
    for line in ins:
        new = re.split(r'\t+', line.rstrip('\t'))
        labels.append(new[1])
        label=new[1]
        array.append(new[2])
del labels[0]
labels = list(map(int, labels))
Y1 = labels[-100:]
print('real label')
print(Y1)
totalvotes=list(map(int,totalvotes))
print("TOTAL VOTE RESULT")
print(totalvotes)
print('TOTAL ACCURACY')
right=0
wrong=0
for i in range(0,count):
    if Y1[i] == totalvotes[i]:
        right+=1
    else :
        wrong +=1     
print(right)
print('TOTAL ERROR')
print(wrong)
tweets=array[-100:]
file = open("predictions-taskA.txt","w") 
#naive1= list(map(int, naive1))
for i in range(0,count):
  #file.write("%d " %totalvotes[i])
  file.write("%d " %totalvotes[i])
  file.write(tweets[i])
file.close()
