# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:06:05 2018

@author: Sherin
"""
import numpy 
from sklearn.naive_bayes import GaussianNB
semantic=[]
bagofwords=[]
wordembedding=[]
labels = []
count=100-1
def readdata(filename):
 mynumber = []
 with open(filename) as f:
    for line in f:
        mynumber.append([n for n in line.strip().split('\n')])
 return mynumber
wordembed=readdata("wordembedfeatures.txt")
sem=readdata("semtfeatures.txt")
label=readdata("label.txt")
bag=readdata("bagfeatures.txt")
y1=numpy.array(wordembed)
y2=numpy.array(sem)
y3=numpy.array(bag)
vectordata=[]
colomns=500 + 3+ 100;
for i in range(0,3832):
     vectordata.extend(y1[i])
     vectordata.extend(y2[i])
     vectordata.extend(y3[i])
colom=len(y1[i])+len(y2[i])+len(y3[i])
print("colom")
print(colom)
vectordata=numpy.ravel(vectordata)
flat_list=[]
print(vectordata)
for sublist in vectordata:
    for item in sublist:
        if item !=',' or item !=' ':
          flat_list.append(item)

print(flat_list)

vectordata=numpy.reshape(flat_list,(3834,colomns))
X = vectordata[:3700]
Y = label[:3700]
X1 = vectordata[-100:]
Y1 = label[-100:]
clf = GaussianNB()
clf.fit(X, Y)
y_pred = clf.fit(X, Y).predict(X1)
print('the error of naive bayes classification')
print((Y1 != y_pred).sum())
print('the accuracy of naive bayes  classification')
print((Y1 == y_pred).sum())
