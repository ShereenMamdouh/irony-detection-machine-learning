

# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:57:29 2017

@author: Micro Systems
"""
import collections
import re
import operator
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
stemmer = PorterStemmer()
lmtzr = WordNetLemmatizer()


def find_features(tweets):
    words = set(tweets)
    features = {}
    for w in word_features:
        features[w] = (w in words)

    return features

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def lem_tokens(tokens, lmtzr):
    lemmed = []
    for item in tokens:
        lemmed.append(lmtzr.lemmatize(item))
    return lemmed

def join_array(array):
    array2 = []
    for word in array:
        array2.append(' '.join(word))
    return array2


def bag_of_words(array):
    vectorizer = CountVectorizer()
    xarray = []
    X = vectorizer.fit_transform(array) 
    xarray = X.toarray()
    y = [sum(row[i] for row in xarray) for i in range(len(xarray[0]))]
    return y, vectorizer.get_feature_names()

tweets = []
index = []
labels = []

stop_words = set(stopwords.words('english'))
stop_words.add(',')
stop_words.add('.')
stop_words.add('...')
stop_words.add('..')
stop_words.add("'s")



with open("SemEval2018-T3-train-taskA.txt", encoding="utf8") as ins:
    array = []
    for line in ins:
        array.append(line)
        new = re.split(r'\t+', line.rstrip('\t'))
        index.append(new[0])
        labels.append(new[1])
        newString =new[2].rstrip('\n')
        newString = re.sub(r"http\S+", "", newString)
        newString = re.sub(r"@\S+ ", "", newString)
        newString = re.sub(r"#", "", newString)
        newString = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', newString)
        newString = re.sub('([a-z0-9])([A-Z])', r'\1 \2', newString).lower()
        firstString = newString
        lastString = word_tokenize(firstString)
        filtered_sentence = [w for w in lastString if not w in stop_words]
        stems = stem_tokens(filtered_sentence, stemmer)
        lems = lem_tokens(filtered_sentence, lmtzr)
        tweets.append(lems)
        
        
del index[0]
index = list(map(int, index))
del labels[0]
labels = list(map(int, labels))
del tweets[0]

#print(array[3])
#print(index[2])
#print(labels[2])
#print(tweets[2])

ironicTweets = []
new_ironicTweets = []
index_i = []

nonTweets = []
new_nonTweets = []
index_n = []

i=0 

for word in tweets:
    i = i+1
    if labels[i-1] == 1:
        ironicTweets.append(word)
        index_i.append(i)
    else:
        nonTweets.append(word)
        index_n.append(i)


new_ironicTweets = join_array(ironicTweets)
new_nonTweets = join_array(nonTweets)
new_tweets = join_array(tweets)


#print(index_i[2])
#print(ironicTweets[2])


#### Ironic hashmap ####

y,x = bag_of_words(new_ironicTweets)
#returns number of occurrences "y" and words referencing them "x"

hashmap = dict(zip(x, y))
#connects x and y in a hashmap (dictionary)
sorted_hash = sorted(hashmap.items(), key=operator.itemgetter(1))
word_features = [x[0] for x in sorted_hash[-250:]]


#sorting the hashmap for our own knowledge purposes

#prints the most repeated word in ironic tweets

#### non-Ironic hashmap ####

y2,x2 = bag_of_words(new_nonTweets)

hashmap2 = dict(zip(x2, y2))
sorted_hash2 = sorted(hashmap2.items(), key=operator.itemgetter(1))
word_features2 = [x[0] for x in sorted_hash2[-250:]]

#prints the most repeated word in non-ironic tweets

#### All-tweets hashmap ####

y3,x3 = bag_of_words(new_tweets)

hashmap3 = dict(zip(x3, y3))
sorted_hash3 = sorted(hashmap3.items(), key=operator.itemgetter(1))
word_features3 = [x[0] for x in sorted_hash3[-500:]]

#prints the most repeated word in all tweets

word_features = sum([word_features, word_features2], [])


def find_features(document):
    words = set(document)
    features = {}
    feature = []
    for w in word_features:
        features[w] = (w in words)
        
        if (w in words):
            feature.append(1)
        else:
            feature.append(0)
    
    

    return (feature)

feature = []

for tweet in tweets:
    feature.append(find_features(tweet))
    

X = feature
Y = labels
X1 = feature
Y1 = labels
clf = GaussianNB()
clf.fit(X, Y)
y_pred = clf.fit(X, Y).predict(X1)

print("results of bag of words")

print('the error of naive bayes classification')
print((Y1 != y_pred).sum()/len(X))
print('the accuracy of naive bayes  classification')
print((Y1 == y_pred).sum()/len(X))
file = open("naive-bag-final.txt","w") 
for item in y_pred:
  file.write("%f" %item)
  file.write("_")
file.close() 
