# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 21:29:55 2017

@author: Sherin
"""

import re
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from gensim.models import Word2Vec 
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
tweets = []
index = []
labels = []
        
        
stemmer = PorterStemmer()
lmtzr = WordNetLemmatizer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def wordtovec(trainingdata):
   model = Word2Vec(trainingdata, min_count=1)
   model.save('model.bin')
   #print(model)
   return model
    

def lem_tokens(tokens, lmtzr):
    lemmed = []
    for item in tokens:
        lemmed.append(lmtzr.lemmatize(item))
    return lemmed
def zerolistmaker(n):
    listofzeros = [0] * n
    return listofzeros

def sentencevector(sentence):
    vectormapping=[]
    trainvector=zerolistmaker(100)
    k=1
    for word in sentence:
        if word in list(model.wv.vocab):
         vectormapping=model[word]
         trainvector+=vectormapping
        k+=1
    trainvector[:] = [x / k for x in trainvector]
    return trainvector
stop_words = set(stopwords.words('english'))
stop_words.add(',')
stop_words.add('.')
stop_words.add('...')
stop_words.add('..')
stop_words.add("'s")
with open("SemEval2018-T3-train-taskA.txt", encoding="utf8") as ins:
    tweets = []
    for line in ins:
        new = re.split(r'\t+', line.rstrip('\t'))
        newString =new[2].rstrip('\n')
        newString = re.sub(r"http\S+", "", newString)
        newString = re.sub(r"@\S+ ", "", newString)
        newString = re.sub(r"#", "", newString)
        newString = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', newString)
        newString = re.sub('([a-z0-9])([A-Z])', r'\1 \2', newString).lower()
        firstString = newString
        data = word_tokenize(firstString)
        stems = stem_tokens(data, stemmer)
        lems = lem_tokens(data, lmtzr)
        tweets.append(lems)
model=wordtovec(tweets)

with open("SemEval2018-T3-train-taskA.txt", encoding="utf8") as ins:
    array = []
    X=[]
    for line in ins:
        new = re.split(r'\t+', line.rstrip('\t'))
        index.append(new[0])
        labels.append(new[1])
        label=new[1]
        newString =new[2].rstrip('\n')
        newString = re.sub(r"http\S+", "", newString)
        newString = re.sub(r"@\S+ ", "", newString)
        newString = re.sub(r"#", "", newString)
        newString = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', newString)
        newString = re.sub('([a-z0-9])([A-Z])', r'\1 \2', newString).lower()
        lastString = word_tokenize(newString)
        filtered_sentence = [w for w in lastString if not w in stop_words]
        stems = stem_tokens(filtered_sentence, stemmer)
        lems = lem_tokens(filtered_sentence, lmtzr)
        embeddedvector=sentencevector(lems)
        array.extend(embeddedvector)
print(len(tweets))
print(len(embeddedvector))
data= np.reshape(array, (len(tweets), len(embeddedvector)))      
del index[0]
index = list(map(int, index))
del labels[0]
labels = list(map(int, labels))
del array[0]
X = data[:3700]
Y = labels[:3700]
X1 = data[-100:]
Y1 = labels[-100:]
clf = GaussianNB()
print("results of word embedding")
y_pred = clf.fit(X, Y).predict(X1)
print('the error of naive bayes classification')
print((Y1 != y_pred).sum())
print('the accuracy of naive bayes  classification')
print((Y1 == y_pred).sum())
file = open("naive-emb.txt","w") 
for item in y_pred:
  file.write("%f" %item)
  file.write("_")
file.close() 
clf1 = svm.SVC()
y_pred = clf1.fit(X, Y).predict(X1)
print('the error of svm classification')
print((Y1 != y_pred).sum())
print('the accuracy of svm  classification')
print((Y1 == y_pred).sum())
file = open("svm-emb.txt","w") 
for item in y_pred:
  file.write("%f" %item)
  file.write("_")
file.close() 
clf = tree.DecisionTreeClassifier()
y_pred= clf.fit(X, Y).predict(X1)
print('the error of svm classification')
print((Y1 != y_pred).sum())
print('the accuracy of svm  classification')
print((Y1 == y_pred).sum())
file = open("trees-emb.txt","w") 
for item in y_pred:
  file.write("%f" %item)
  file.write("_")
file.close() 
model = KNeighborsClassifier(n_neighbors=1);
model.fit(X, Y);
Y_pred=model.predict(X1);
accuracy=(Y1 ==Y_pred).sum()
print('the error of nearest neighboor classification')
print(100-accuracy)
print('the accuracy of nearest neighboor  classification')
print(accuracy)
file = open("nearest.txt","w") 
for item in Y_pred:
  file.write("%f" %item)
  file.write("_")
file.close()
   
