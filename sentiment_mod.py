import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI #inherit
from statistics import mode

from nltk.tokenize import word_tokenize
import codecs

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

docs_file = open("19-pickled/documents_.pickle","rb")
documents = pickle.load(docs_file)
docs_file.close()

word_features_file = open("19-pickled/word_features5k_.pickle","rb")
word_features = pickle.load(word_features_file)
word_features_file.close()

def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = bool(w in words)

	return features

featuresets_file = open("19-pickled/featuresets_.pickle","rb")
featuresets = pickle.load(featuresets_file)
featuresets_file.close()

random.shuffle(featuresets)
print(len(featuresets))

training_set = featuresets[:10000]
testing_set = featuresets[10000:]


open_file = open("19-pickled/naive_bayes_.pickle","rb")
classifier = pickle.load(open_file)
open_file.close()

open_file = open("19-pickled/mnb_.pickle","rb")
MNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("19-pickled/bernoulli_.pickle","rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()

open_file = open("19-pickled/logisticregression_.pickle","rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

open_file = open("19-pickled/sgd_.pickle","rb")
SGDClassifier_classifier = pickle.load(open_file)
open_file.close()

open_file = open("19-pickled/linearsvc_.pickle","rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()

open_file = open("19-pickled/nu_svc_.pickle","rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()

voted_classifier = VoteClassifier(classifier, 
									MNB_classifier, 
									BernoulliNB_classifier, 
									LogisticRegression_classifier, 
									SGDClassifier_classifier, 
									LinearSVC_classifier, 
									NuSVC_classifier)
print("voted_classifier accuracy:", nltk.classify.accuracy(voted_classifier, testing_set))

def sentiment(text):
	featset = find_features(text)
	return voted_classifier.classify(featset), voted_classifier.confidence(featset)



