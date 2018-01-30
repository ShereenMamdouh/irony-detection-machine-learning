# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 19:24:39 2018

@author: Sherin
"""
import re
import operator
import numpy as np

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec 
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
count=0


with open("SemEval2018-T3-train-test-taskB.txt", encoding="utf8") as ins:
    array = []
    for line in ins:
        array.append(line)
        new = re.split(r'\t+', line.rstrip('\t'))
        if(len(new)!=3):
            break
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
print('done with bag of words')
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
    if len(sentencePart)!=0 :
      probneg=len(neg_word_list)/len(sentencePart)
      probpos=len(pos_word_list)/len(sentencePart)
      probnue=len(neu_word_list)/len(sentencePart)
    else: 
      probpos=probneg=probnue=0
    return probpos,probneg,probnue
stop_words = set(stopwords.words('english'))

stop_words.add(',')
stop_words.add('.')
stop_words.add('...')
stop_words.add('..')
stop_words.add("'s")

removeFromStopWords(keep)

totalironic=0;
ironicaccuracy=0;
SEMANTICVECTOR=[]
with open("SemEval2018-T3-train-test-taskB.txt", encoding="utf8") as ins:
    tweets = []
    for line in ins:
        new = re.split(r'\t+', line.rstrip('\t'))
        if(len(new)!=3):
            break
        newString =new[2].rstrip('\n')
        newString = re.sub(r"http\S+", "", newString)
        newString = re.sub(r"@\S+ ", "", newString)
        newString = re.sub(r"#", "", newString)
        newString = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', newString)
        newString = re.sub('([a-z0-9])([A-Z])', r'\1 \2', newString).lower()
        firstString = newString
        data = word_tokenize(firstString)
        tweets.append(firstString)  
with open("SemEval2018-T3-train-test-taskB.txt", encoding="utf8") as ins:
    array = []
    probablities=[]
    for line in ins:       
        new = re.split(r'\t+', line.rstrip('\t'))
        if(len(new)!=3):
            break
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
        for i in range(len(conjunctions)):
            if conjunctions[i] in firstString:
                split = firstString.split(conjunctions[i])
                lastStringPart1 = word_tokenize(split[0])
                lastStringPart2 = word_tokenize(split[1])              
                filtered_sentence1 = [w for w in lastStringPart1 if not w in stop_words]
                filtered_sentence2 = [w for w in lastStringPart2 if not w in stop_words] 
                pos1,neg1,nue1=sentimentAnalysis(filtered_sentence1)
                pos2,neg2,nue2=sentimentAnalysis(filtered_sentence2)
                probablities.append([(pos1+pos2)/2,(neg1+neg2)/2,(nue1+nue2)/2])
                
                if  neg1>pos1 and pos2>neg2 or max(neg1,neg2)>0.5 or max(pos1,pos2)>0.5:
                 totalironic+=1
                 if label=='1':
                   ironicaccuracy+=1 
                elif neg2>pos2 and pos1>neg1 or max(neg1,neg2)>0.5 or max(pos1,pos2)>0.5 :
                 totalironic+=1
                 if label=='1':
                   ironicaccuracy+=1  
                else :
                 if label=='0':
                   ironicaccuracy+=1   
                 
                break
            else: 
                lastString = word_tokenize(firstString)
                filtered_sentence = [w for w in lastString if not w in stop_words]
                pos,neg,nue=sentimentAnalysis(filtered_sentence)
                probablities.append([pos,neg,nue])
                if pos < neg or neg >0.5 or pos>0.5:
                 totalironic+=1
                 if label=='1':
                   ironicaccuracy+=1                   
                else :
                 if label=='0':
                   ironicaccuracy+=1 
                break
tweets = []
index = []
labels = []
print('done with semantic')
        
def wordtovec(trainingdata):
   model = Word2Vec(trainingdata, min_count=1)
   model.save('model.bin')
   return model
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
with open("SemEval2018-T3-train-test-taskB.txt", encoding="utf8") as ins:
    tweets = []
    for line in ins:
        new = re.split(r'\t+', line.rstrip('\t'))
        if(len(new)!=3):
            break
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

with open("SemEval2018-T3-train-test-taskB.txt", encoding="utf8") as ins:
    array = []
    X=[]
    for line in ins:
        new = re.split(r'\t+', line.rstrip('\t'))
        
        if(len(new)!=3):
            break
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
data= np.reshape(array, (len(tweets), len(embeddedvector)))      
del index[0]
index = list(map(int, index))
del labels[0]
labels = list(map(int, labels))
del array[0]
print('done with wordembedding')
VECTORDATA=[]

probablities = probablities[:-1]
data = data[:-1]
VECTORDATA = np.concatenate((data,probablities),axis=1)
VECTORDATA = np.concatenate((feature,VECTORDATA),axis=1)
print(np.shape(VECTORDATA))

X = VECTORDATA[:3800]
Y = labels[:3800]
X1 = VECTORDATA[-784:]
Y1 = labels[-784:]
clf = GaussianNB()
clf.fit(X, Y)
y_pred = clf.fit(X, Y).predict(X1)
print("results ")
#print('the error of naive bayes classification')
#print((((Y1 != y_pred).sum())/784)*100)
#print('the accuracy of naive bayes  classification')
#print((((Y1 == y_pred).sum()) /784)*100)
clf1 = svm.SVC()
y_pred = clf1.fit(X, Y).predict(X1)
file = open("result.txt","w") 
for item in y_pred:
  file.write("%d" %item)
  file.write('\n')
file.close()
#print('the error of svm classification')
#print((((Y1 != y_pred).sum())/784)*100)
#print('the accuracy of svm  classification')
#print((((Y1 == y_pred).sum())/784)*100)
clf = tree.DecisionTreeClassifier()
y_pred= clf.fit(X, Y).predict(X1)
file = open("result1.txt","w") 
for item in y_pred:
  file.write("%d" %item)
  file.write('\n')
file.close()
#print('the error of trees classification')
#print((((Y1 != y_pred).sum())/784)*100)
#print('the accuracy of trees classification')
#print((((Y1 == y_pred).sum())/784)*100)
model = KNeighborsClassifier(n_neighbors=1);
model.fit(X, Y);
Y_pred=model.predict(X1);
file = open("result2.txt","w") 
for item in Y_pred:
  file.write("%d" %item)
  file.write('\n')
file.close()
accuracy=((((Y1 ==Y_pred).sum())/784)*100)
#print('the error of nearest neighboor classification')
#print(100-accuracy)
#print('the accuracy of nearest neighboor  classification')
#print(accuracy)

