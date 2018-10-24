#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals, division, print_function
import sys, imp
imp.reload(sys)
sys.setdefaultencoding('utf8')
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
import codecs

import random
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_validate
from sklearn import metrics

dataset = {}
labels = set()
featurenames = {}

with codecs.open("./labeled_data.txt", "rt", encoding="utf-8") as infile:
    for line_index, line in enumerate(infile):
        if line.strip() == '':
            continue
        try:
            pmid, original, classname, features = line.split("\t")
            pmid = int(float(pmid))
            features = features.strip().split(" ")
            #print(pmid, classname, features)
            labels.add(classname)
            if not pmid in dataset:
                dataset[pmid] = {}

            # create features as dicts (featurename = 1.0) for DictVectorizer later on
            dataset[pmid][classname] = {}
            for feature in features:
                dataset[pmid][classname][feature] = 1.0
                featurenames[feature] = 1.0
        except Exception as e:
            print("Malformed '%s' (#%s) --> %s" % (line, line_index + 1, e))

unmatched = 0
for pmid in dataset:
    data = dataset[pmid]
    prevfeatures = None
    for classname in data:
        features = data[classname].keys()
        if not prevfeatures is None:
            if not features == prevfeatures:
                print(pmid, classname, features == prevfeatures)
        prevfeatures = features


# determine which classifiers we need to train based on the labels
classifiers = {}
for label in labels:
    labelprefix = label.split("-")[0]
    if not labelprefix in classifiers:
        classifiers[labelprefix] = set()
    classifiers[labelprefix].add(label)
#print(classifiers)

# create learning algorithm instances to try on the data
try_algorithms = {}

try_algorithms['SVC'] = svm.SVC(kernel='linear', C=1, random_state=0)
try_algorithms['NB'] = MultinomialNB()
try_algorithms['LReg'] = LogisticRegression()
try_algorithms['SGD'] = SGDClassifier()

for classifier, labels in classifiers.items():
    print("training classifier", classifier, "for labels", ", ".join(labels))
    # gather data with the labels in this classifier
    classifier_labels = []
    classifier_data = []
    for pmid in dataset:
        for label in dataset[pmid]:
            if not label in labels:
                # not relevant for current classifier
                continue
            features = dataset[pmid][label]
            classifier_labels.append(label)
            classifier_data.append(features)
    random.shuffle(classifier_data)

    # encode features
    # (create integer indices for all feature names)
    dv = DictVectorizer(sparse=False)
    classifier_data_transformed = dv.fit_transform(classifier_data)

    # encode labels
    # (create integer encodings for class labels)
    le = LabelEncoder()
    le.fit(list(labels))
    classifier_labels_transformed = le.fit_transform(classifier_labels)

    # the _transformed arrays now contain only numeric information

    # now we can train and evaluate a classifier (in 5-fold cross validation)
    for algo, clf in try_algorithms.items():
        print("training", algo, "on", classifier)
        scoring = ['precision_macro', 'recall_macro', 'f1_macro', 'accuracy']
        scores = cross_validate(clf, classifier_data_transformed, classifier_labels_transformed, scoring=scoring, cv=5, return_train_score=False)

        # output scores and timing information:
        for k in sorted(scores.keys()):
            print("\t%s %s %s: %0.2f (+/- %0.2f)" % (classifier, algo, k, scores[k].mean(), scores[k].std() * 2))

