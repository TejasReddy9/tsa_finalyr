---
layout: default
---

# Overview
Using online streaming tweets-data from Twitter API, this is an attempt to trace out the most appropriate sentiment related to the topic of interest. I've started with Naive Bayes Classifier and ended up by creating my own classifier by setting votes on which classifier to be considered more. My classifier is based on many classifiers namely Basic Naive Bayes, Multinomial Naive Bayes, Bernoulli Naive Bayes, Logistic Regression, Stochastic Gradient Descent and Linear SVM classifiers combined together.
It basically creates votes for each classification and returns the result of that classifier whose statistical mode of votes is maximum. 

## Requirements
Install Python3.x, and for all first install these dependencies using pip - tweepy, json, StreamListner, nltk, scikit-learn, sklearn, scipy, pandas, numpy, matplotlib, pickle.
```
pip3 install dependency_name
```

For nltk, we need to download the corpus. This is done as follows.. Enter 'all' when prompted.
```python
import nltk
nltk.download_shell()
```

For getting live data from twitter, go to the [twitter apps site](https://apps.twiiter.com). Attach mobile number to your twitter account and create app. You'll be given a consumer key and consumer secret key. Go down and click on authorise app. You'll be given an auth key and auth secret key. Keep them confidential.

## Module created for Sentiment Analysis
*   First I've used movie-reviews dataset from nltk corpora is used for training our model. Later I found a better dataset, lets use some data from reddit comments(pretty big and famous). I've attached them in the repo.
```python
short_pos = codecs.open("positive.txt","r",encoding="latin2").read()
short_neg = codecs.open("negative.txt","r",encoding="latin2").read()
```

*   Let's pile up all the data into a documents list. Additionally, we can filter these words which we wanted in dictionary w.r.t. their part of speech tag using nltk. For example, starts with 'J' only for adjectives. Documents is the list of tuples. One is the sentence, other is the sentiment.
```python
documents = []
all_words = []
allowed_words_types = ["J"]
for r in short_pos.split('\n'):
	documents.append((r,"pos"))
	words = word_tokenize(r)
	pos = nltk.pos_tag(words)
	for w in pos:
            if w[1][0] in allowed_words_types:
	        all_words.append(w[0].lower())
for r in short_neg.split('\n'):
	documents.append((r,"neg"))
	words = word_tokenize(r)
	pos = nltk.pos_tag(words)
	for w in pos:
	    if w[1][0] in allowed_words_types:
		all_words.append(w[0].lower())
```

*   Pickle everything meanwhile, it saves running time for the program as it had to load the whole data each time you run.

*   Then find frequency distribution of all_words(list of sentences). This gives us the idea of the most frequently occuring words. Let's take the top 5k or 7k data of them to make word features.
```python
all_words = nltk.FreqDist(all_words) #frequencies
word_features = list(all_words.keys())[:5000]
```

*   For finding features, for sentence in documents, we replace it with a dictionary of features of the sentence. Hence we get featureset. For each word in word features, if the word exists in the tokenized words, then features of it is set True. For each sentence, word features list is labelled. Shuffle them.
```python
def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
	    features[w] = bool(w in words)
	return features
featuresets = [(find_features(rev), category) for (rev, category) in documents]
random.shuffle(featuresets)
```

*   Train various models. Change the training data and testing data quantity if you want to.
```python
training_set = featuresets[:10000]
testing_set = featuresets[10000:]
classifier = nltk.NaiveBayesClassifier.train(training_set)
# classifier = SklearnClassifier(MultinomialNB()).train(training_set)
# classifier = SklearnClassifier(BernoulliNB()).train(training_set)
# classifier = SklearnClassifier(LogisticRegression()).train(training_set)
# classifier = SklearnClassifier(SGDClassifier()).train(training_set)
# classifier = SklearnClassifier(LinearSVC()).train(training_set)
# classifier = SklearnClassifier(NuSVC()).train(training_set)
print("Accuracy:", nltk.classify.accuracy(classifier, testing_set))
```

*   Let's create our own classifier based on all these classifiers, by preferring that model which gets most number of votes.
```python
class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
	    self._classifiers = classifiers
	def classify(self, features):
	    votes = [c.classify(features) for c in self._classifiers]
	    return mode(votes)
	def confidence(self, features):
	    votes = [c.classify(features) for c in self._classifiers]
	    choice_votes = votes.count(mode(votes))
	    conf = choice_votes / len(votes)
	    return conf 
voted_classifier = VoteClassifier(classifier, MNB_classifier, BernoulliNB_classifier, LogisticRegression_classifier, 
				SGDClassifier_classifier, LinearSVC_classifier, NuSVC_classifier)
print("voted_classifier accuracy:", nltk.classify.accuracy(voted_classifier, testing_set))
```

*   A function for determining expected sentiment.
```python
def sentiment(text):
	featset = find_features(text)
	return voted_classifier.classify(featset)
```

## Results from predictions
Based on weighted mean absolute error(WMAE), points scored in Kaggle is 11126.41686. Please feel free to download, fork, send pull requests and comment on this project.
