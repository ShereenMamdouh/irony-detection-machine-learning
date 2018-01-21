# irony detection-machine learning
Main objective is to classify a tweet as Ironic or Non-Ironic. This covers natural language processing and classification. 
Our Course of Action
In general, irony is a tricky subject that even humans sometimes fail to detect it in a conversation or a sentence, but still, it must have a certain pattern or there exist some words that hints that irony exists. We examined many sentences that contain irony, and we deduced the following.
1.	A negation word, like not and no, exists in the sentence.

2.	The sentence usually contains two emotions that contradict each other.
We decided to split the sentence on the conjunction of the sentence, if it exists, and evaluate the emotion of each of each part of the sentence.
Examples of conjunctions are which, and, but,..etc.

3.	The sentence usually has a negative sentiment feeling.
Assignment's Quick Steps
1.	Text Preprocessing
2.	Train Models
3.	Results 
The Dataset
The used dataset in this assignment is provided the one provided in SemEval-2018 task 3. It consists of 3842 tweets as a total. The tweets were picked by searching Twitter for tweets that contains hashtags #irony, #sarcasm and #not. Later, the tweets were refined by three students in linguistics and second-language speakers of English. 
%%%%%%%%%%%%%%%
Format
The dataset consists of 3842 tweets as a total. 
?	604 tweets are classified as Non-Ironic.
?	2396 tweets are classified as Ironic.

There are 3 attributes, separated by 1 tab, that are in the following order:
1.	Index
2.	Class (either 0 or 1)
1: Ironic
0: Non-Ironic
3.	Tweet


Reading the Dataset
According to the format above, we split the dataset into 3 vectors.
?	Index – contains index of each tweet
?	Labels – contains 0/1 (Ironic/Non-Ironic) of each tweet
?	Tweets – contains each tweet
For every line in text file of dataset, split at tabs and save each part of the tweet into correct vector.
%%%%%%%%%%%%%%%%%%%%%%%%
Text Preprocessing
DATA FILTERING
To generate good results and lessen the number of unneeded computations, the tweets are filtered according to certain criteria. 
Stop Words
Generally, in Natural Language Processing, the first step is to always filter the data and remove any insignificant and redundant words. There are known words, called Stop Words, that are always removed to enhance the performance.
Below is a set of some of the Stop Words.
For the objective of this assignment, irony detection in tweets, we removed some words from the Stop Words sets because they are significant in detection the irony. We removed any negating words and conjunctions.

Tokenization
Second step in Natural Language Processing is Tokenization.
Tokenization is the process of splitting sentences into single words or chunks, group of words. The tokenization applied in this assignment is splitting the sentence into single words.
Using nltk library, below is the snippet of code that applies tokenization on tweets.
Lemmatization
Lemmatization is the process of getting the root of a word. Lemmatization takes into consideration the morphological analysis of words. A Lemma is the same for variations of a word, therefore; it reduces noise.
We also have sentences filtered from mentions hash tags and URL’S and numbers
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



EXTRACTING FEATURES
Bag of Words
First, we created 3 arrays. the first array for the words in ironic tweets, the second array for the words in non ironic tweets and the final array for words in all tweets.
 Second, we calculated the number of repetitions of every word in the ironic tweets array across all ironic tweets. We repeated the same step for every word in the non ironic tweets across all non ironic tweets. 
Third, we extracted the most common words (with highest frequency) across both tweets to eliminate them from our processing to the data - since they won’t be effective in determining if a tweet is ironic or not. 
Fourth, we created hashmaps for the words as 'key' attribute and their frequency value as 'value' attribute - one hashmap for words in ironic tweets, another for words in non ironic and the last one for the common ones.
Fifth, we sorted the hashmaps for easy acquiring of the words with highest frequencies.
Finally, we added the hashmaps as another feature for the data processing procedure.
 
 

Sentiment Analysis(FROM OUR SEARCH)
We used sentiment analysis to help detect irony in a tweet. We used polarity feature of a word to determine if the feelings in the tweet changed 180 degrees. For the purpose of this assignment, we removed some of the words from the Stop Words set as mentioned above. We also didn’t apply lemmatization in this feature because it will affect the performance of the polarity test. We also handle emojis and negation words  in the tweets since they contribute to the polarity of the sentence. 
Below are the steps we perform.
1.	Split, into two parts, the tweet on a conjunction from list below. We gathered all the available conjunctions in English Grammar. We handle all the conjunctions except the ones that consist of more than one word like not only...but also,..etc.
2.	Tokenize each part of the tweet.
3.	Remove Stop Words from each part of the tweet.
4.	Evaluate polarity of each word of each part of the sentence


The sentimentAnalysis method takes part of the tweet as input, and using SentimentIntensityAnalyzer from nltk library and the sets of the emojis, the polarity of each word is returned as either positive, negative or neutral. The threshold 0.3 was tuned by running the methods with several values and picking the right value that returned each word to its appropriate polarity.
To calculate the overall polarity of one part of the tweet, we search for the polarity category that has highest number of words and return it as the overall polarity of this tweet. Overall, we calculate the probability of each category of the polarity and pick the highest.
5. Compare the polarities of the sentence parts. There are three conditions the tweet will be classified as Ironic.
1.	Sentence contains contradicting polarity, positive and negative.
2.	Total negative polarity of the whole tweet is greater than 0.3.
3.	Total positive polarity of the whole tweet  is greater than 0.3.
The value 0.3 was chosen after some testing and tuning.

Word Embedding(FROM OUR SEARCH)
First, we created a model with the training data to act like a dictionary for the coming processing. Second, we used every word in every tweet and passed it to the model - which as a result, returned an equivalent numerical vector to the word with length = 100. 
Third, we added all the vectors of the words per tweet and divided this sum by their number. Thus, we acquired a numerical representation of length = 100 for every tweet. 
Fourth, we appended all those vectors of all tweets.
Finally, we pass the resulting appended vectors of all tweets to the classifier.
 
 
This is the case only with words existing in our dictionary.  If the word didn’t exist in the dictionary we made beforehand, a vector of length = 0 is returned.
Supervised Learning
Using skleran library, four models are used to test the features. We used two voting systems. The testing steps are as the following.
1.	Each feature is tested on each model. Using a voting system, the best model is picked. The result is 4 possible classifications.
2.	Using another voting system, the model with the highest accuracy is picked. If two or more models return the same accuracy, one model is picked randomly from them.
%%%%%%%%%%%%%%%%%%%%%
The four models are listed below.
1.	Naive Bayes Classifier
Naive Bayes classifier is a probabilistic classifier that assumes independence between features.

2.	SVM
SVM is a supervised learning algorithm that aims to increase gap between data to divide the data into two category.

3.	Random Forests
Random Forests is an algorithm that generates many decision trees.

4.	K-Nearest Neighbor Classifier
K-Nearest Neighbor Classifier is an algorithm that picks label from k closest data according to a similarity measure.
 After some tuning, k = 1 generated best results for all the features.
%%%%%%%%%%%%%
Bag of Words
Naive Bayes Classifier results in best accuracy.
Semantic Analysis
K-nearest Neighbor Classifier results in best accuracy.
Word Embedding
Decision Trees results in best accuracy.
%%%%%%%%%%%%%%
Python Libraries
?	Numpy
?	sklearn
?	nltk
?	re
?	gensim

%%%%%%%%%%%% extra work %%%%%%%%%%
all results as in label are saved in .txt from each feature and we make a voting system in all features on the labels provided so far the best accuracy is almost 70 % by naive bayes in case of bag of words as the problem defination is 80% ironic so the problem isn't proporly defined 
the next step is to work on sentence 2 vec and LMST and also to combine all data from features into 1 big matrix and check the results where we train data on all features together it's still underimplementation u can check the code at combine.py
