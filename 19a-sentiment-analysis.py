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

short_pos = codecs.open("positive.txt","r",encoding="latin2").read()
short_neg = codecs.open("negative.txt","r",encoding="latin2").read()

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

save_docs = open("documents_.pickle","wb")
pickle.dump(documents, save_docs)
save_docs.close()

all_words = nltk.FreqDist(all_words) #frequencies

save_docs = open("words_.pickle","wb")
pickle.dump(all_words, save_docs)
save_docs.close()

word_features = list(all_words.keys())[:5000]

save_docs = open("word_features_.pickle","wb")
pickle.dump(word_features, save_docs)
save_docs.close()

def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = bool(w in words)

	return features

featuresets = [(find_features(rev), category) for (rev, category) in documents]

save_docs = open("featuresets_.pickle","wb")
pickle.dump(featuresets, save_docs)
save_docs.close()

random.shuffle(featuresets)

training_set = featuresets[:10000]
testing_set = featuresets[10000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Original Naive Bayes Algo accuracy:", nltk.classify.accuracy(classifier, testing_set))
# classifier.show_most_informative_features(15)

save_classifier = open("naive_bayes_.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()


MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier Naive Bayes Algo accuracy:", nltk.classify.accuracy(MNB_classifier, testing_set))

save_classifier = open("mnb_.pickle","wb")
pickle.dump(MNB_classifier, save_classifier)
save_classifier.close()

# GaussianNB_classifier = SklearnClassifier(GaussianNB())
# GaussianNB_classifier.train(training_set)
# print("GaussianNB_classifier Naive Bayes Algo accuracy:", nltk.classify.accuracy(GaussianNB_classifier, testing_set))

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print("BernoulliNB_classifier Naive Bayes Algo accuracy:", nltk.classify.accuracy(BernoulliNB_classifier, testing_set))

save_classifier = open("bernoulli_.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier Linear model accuracy:", nltk.classify.accuracy(LogisticRegression_classifier, testing_set))

save_classifier = open("logisticregression_.pickle","wb")
pickle.dump(LogisticRegression_classifier, save_classifier)
save_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print("SGDClassifier_classifier Linear model accuracy:", nltk.classify.accuracy(SGDClassifier_classifier, testing_set))

save_classifier = open("sgd_.pickle","wb")
pickle.dump(SGDClassifier_classifier, save_classifier)
save_classifier.close()

# SVC_classifier = SklearnClassifier(SVC())
# SVC_classifier.train(training_set)
# print("SVC_classifier SVM accuracy:", nltk.classify.accuracy(SVC_classifier, testing_set))

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier SVM accuracy:", nltk.classify.accuracy(LinearSVC_classifier, testing_set))

save_classifier = open("linearsvc_.pickle","wb")
pickle.dump(LinearSVC_classifier, save_classifier)
save_classifier.close()

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier SVM accuracy:", nltk.classify.accuracy(NuSVC_classifier, testing_set))

save_classifier = open("nu_svc_.pickle","wb")
pickle.dump(NuSVC_classifier, save_classifier)
save_classifier.close()

voted_classifier = VoteClassifier(classifier, 
									MNB_classifier, 
									BernoulliNB_classifier, 
									LogisticRegression_classifier, 
									SGDClassifier_classifier, 
									LinearSVC_classifier, 
									NuSVC_classifier)
print("voted_classifier accuracy:", nltk.classify.accuracy(voted_classifier, testing_set))

# save_classifier = open("voted_.pickle","wb")
# pickle.dump(voted_classifier, save_classifier)
# save_classifier.close()

def sentiment(text):
	featset = find_features(text)
	return voted_classifier.classify(featset)



