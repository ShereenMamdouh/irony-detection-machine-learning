# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 15:57:29 2017

@author: Sherin
"""
import collections
import re
import numpy as np
from sklearn import tree
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from gensim.models import Word2Vec 
from sklearn.neighbors import KNeighborsClassifier
tweets = []
index = []
labels = []
keep = ['not', 'no', 'for', 'and', 'nor', 'but', 'or', 'so',
        'while', 'if', 'only', 
        'until', 'than', 
         'as', 'after', 'before',
        'by', 'now', 'once',
        'when', 'because','in',
        'why', 'what', 'which', 'who', 
         'how', 'where','just', 'both', 
        'with', 'then']

conjunctions = ['for', 'and', 'nor', 'but', 'or', 'yet', 'so',
        'though', 'although', 'even though', 'while', 'if', 'only if', 'unless',
        'until', 'provided that', 'assuming that', 'even if', 'in case', 'than', 'rather than',
        'whether', 'as much as', 'whereas', 'after', 'as long as', 'as soon as', 'before',
        'by the time', 'now that', 'once', 'since', 'till', 'until',
        'when', 'whenever', 'while', 'because', 'since', 'so that', 'in order',
        'why', 'that', 'what', 'whatever', 'which', 'whichever', 'who', 'whoever',
        'whom', 'whomever', 'whose', 'how', 'as though', 'as if','where', 'wherever',
        'also', 'besides', 'furthermore', 'likewise', 'moreover', 'however', 'nevertheless',
        'nonetheless', 'still', 'conversely', 'instead', 'otherwise', 'rather', 'accordingly',
        'consequently', 'hence', 'meanwhile', 'then', 'therefore', 'thus']

NEGATE = {'ain\'t', 'aren\'t', 'cannot', 'can\'t', 'couldn\'t', 'daren\'t', 'didn\'t', 'doesn\'t',
 'ain\'t', 'aren\'t', 'cant', 'couldn\'t', 'daren\'t', 'didn\'t', 'doesn\'t',
 "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", 'neither',
 "don't", "hadn't", "hasn't", "haven't", "isn't", "mightn't", "mustn't",
 "neednt", "needn't", "never", "none", "nope", "nor", "not", "nothing", "nowhere",
 "oughtnt", "shant", "shouldnt", "uhuh", "wasnt", "werent",
 "oughtn't", "shan't", "shouldn't", "uh-uh", "wasn't", "weren't",
 "without", "wont", "wouldnt", "won't", "wouldn't", "rarely", "seldom", "despite",'!'}

HAPPY = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P'
    , ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])

SAD = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])



def removeFromStopWords(keep):
    for i in range(len(keep)):
        stop_words.remove(keep[i])
        
        
stemmer = PorterStemmer()
lmtzr = WordNetLemmatizer()

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


def sentimentAnalysis(sentencePart):

    
    sid = SentimentIntensityAnalyzer()

    pos_word_list=[]
    neu_word_list=[]
    neg_word_list=[]
    emoji=[]

    for word in sentencePart:

        if (sid.polarity_scores(word)['compound']) >= 0.3 or word in HAPPY:
            pos_word_list.append(word)
        elif (sid.polarity_scores(word)['compound']) <= -0.3 or word in NEGATE or word in SAD:
            neg_word_list.append(word)
        else:
            neu_word_list.append(word)
        if  emoji  or word in {':','=','>',';','X'}:
            emoji.append(word)
            if word in neu_word_list:
              neu_word_list.remove(word)

    emoji=''.join(emoji)
    if emoji in HAPPY :
       pos_word_list.append(emoji) 
    if emoji in SAD:
       neg_word_list.append(emoji)
    #print('Positive :',pos_word_list)        
    #print('Neutral :',neu_word_list)    
    #print('Negative :',neg_word_list)
    vectormapping=[]
    if len(sentencePart)!=0 :
      probneg=len(neg_word_list)/len(sentencePart)
      probpos=len(pos_word_list)/len(sentencePart)
      probnue=len(neu_word_list)/len(sentencePart)
    else: 
      probpos=probneg=probnue=0
    #print(vectormapping)
    return probpos,probneg,probnue
stop_words = set(stopwords.words('english'))

stop_words.add(',')
stop_words.add('.')
stop_words.add('...')
stop_words.add('..')
stop_words.add("'s")

removeFromStopWords(keep)

print(stop_words)
totalironic=0;
ironicaccuracy=0;
SEMANTICVECTOR=[]
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
        tweets.append(firstString)
#model=wordtovec(tweets)
  
with open("SemEval2018-T3-train-taskA.txt", encoding="utf8") as ins:
    array = []
    probablities=[]
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
        firstString = newString
        array.append(firstString)
        #print('firstString')
        #print(firstString)
        #split tweet into two if a conjuction exist
        #for i in range(len(conjunctions)):
        for i in range(len(conjunctions)):
            if conjunctions[i] in firstString:
                
                #print(conjunctions[i])
                split = firstString.split(conjunctions[i])
                #print(split)

                lastStringPart1 = word_tokenize(split[0])
                lastStringPart2 = word_tokenize(split[1])
                
                filtered_sentence1 = [w for w in lastStringPart1 if not w in stop_words]
                filtered_sentence2 = [w for w in lastStringPart2 if not w in stop_words]
                
                #print(filtered_sentence1)
                #print(filtered_sentence2)
                #print('the ironic counter',totalironic)
                #print('the accuracy so far:',ironicaccuracy) 
                pos1,neg1,nue1=sentimentAnalysis(filtered_sentence1)
                pos2,neg2,nue2=sentimentAnalysis(filtered_sentence2)
                #wordtovec=wordtovec([filtered_sentence1,filtered_sentence2])
                probablities.append([(pos1+pos2)/2,(neg1+neg2)/2,(nue1+nue2)/2])
                
                #print('the sentence has conjunction')
                if  neg1>pos1 and pos2>neg2 or max(neg1,neg2)>0.5 or max(pos1,pos2)>0.5:
                 totalironic+=1
                 #print('the sentence classifed ironic with prob',neg1+pos2)
                 if label=='1':
                   ironicaccuracy+=1 
                   #print('the ironic counter accuracy',ironicaccuracy)
                elif neg2>pos2 and pos1>neg1 or max(neg1,neg2)>0.5 or max(pos1,pos2)>0.5 :
                 totalironic+=1
                 #print('the sentence classifed ironic with prob',neg2+pos1) 
                 if label=='1':
                   ironicaccuracy+=1
                   
                else :
                 #print('the sentence isnt ironic')
                 if label=='0':
                   ironicaccuracy+=1   
                 
                break
            else: 
                #print('the ironic counter',totalironic)
                lastString = word_tokenize(firstString)
                filtered_sentence = [w for w in lastString if not w in stop_words]
                pos,neg,nue=sentimentAnalysis(filtered_sentence)
                #wordtovec=wordtovec(filtered_sentence)
                probablities.append([pos,neg,nue])
                if pos < neg or neg >0.5 or pos>0.5:
                 totalironic+=1
                 #print('the sentence classifed ironic with prob',neg) 
                 if label=='1':
                   ironicaccuracy+=1 
                   
                else :
                 #print('the sentence isnt ironic')
                 if label=='0':
                   ironicaccuracy+=1 
                break
#print('ironicaccuracy') 
#print(ironicaccuracy)  
#print(probablities)

del index[0]
index = list(map(int, index))
del labels[0]
labels = list(map(int, labels))
del array[0]
X = probablities[:3700]
Y = labels[:3700]
X1 = probablities[-100:]
Y1 = labels[-100:]
clf = GaussianNB()
y_pred = clf.fit(X, Y).predict(X1)
file = open("naive.txt","w") 
for item in y_pred:
  file.write("%f " %item)
  file.write("_")
file.close() 
print("results ofsemanic analsysis")
print('the error of naive bayes classification')
print((Y1 != y_pred).sum())
print('the accuracy of naive bayes classification')
print((Y1 == y_pred).sum())
clf1 = SVC()
y_pred = clf1.fit(X, Y).predict(X1)
print('the error of svm classification')
print((Y1 != y_pred).sum())
print('the accuracy of svm  classification')
print((Y1 == y_pred).sum())
file = open("svm.txt","w") 
for item in y_pred:
  file.write("%f " %item)
  file.write("_")
file.close() 
clf = tree.DecisionTreeClassifier()
y_pred= clf.fit(X, Y).predict(X1)
print('the error of decision trees  classification')
print((Y1 != y_pred).sum())
print('the accuracy of decision trees  classification')
print((Y1 == y_pred).sum())
file = open("trees.txt","w") 
for item in y_pred:
  file.write("%f  " %item)
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
   
