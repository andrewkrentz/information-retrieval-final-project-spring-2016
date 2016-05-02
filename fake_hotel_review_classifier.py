"""
Andrew Krentz
Information Retrieval
Spring 2016

Final Project
Yelp Data Analysis
fake_hotel_review_classifier.py

Trains and tests a classifier on the fake review data set. Then the model
is used to predict whether some of the Yelp hotel reviews are fake.

The fake review data was collected by Myle Ott for
the following paper and is located at http://myleott.com/op_spam/:

[1] M. Ott, C. Cardie, and J.T. Hancock. 2013. Negative Deceptive Opinion Spam.
In Proceedings of the 2013 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies.
"""

import string
import nltk
import numpy as np
import pandas as pd
import itertools
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

INPUT_FILE = 'data/op_spam.csv'
HOTEL_REVIEWS = 'data/yelp_hotel_reviews.csv'

# Downloads the NLTK work tokenizer model.
nltk.download('punkt')

# Collection of stop words to filter out.
stop_words = stopwords.words('english')

# Map of punctuation characters to be removed from documents during processing.
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)

sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)


def process_review(review):
    # Remove all punctuation and lowercase the string.
    value_without_punctuation = review.translate(string.maketrans("", ""), string.punctuation).lower()

    # Use NLTK word tokenizer to tokenize the string.
    tokens = [x for x in nltk.word_tokenize(value_without_punctuation) if x not in stop_words]

    # Combine all of the tokens back into a single string and add to results.
    return " ".join(tokens)

# Read in CSV data, create train and test sets, and tokenize the review text.
df = pd.read_csv(INPUT_FILE)
train, test = train_test_split(df.copy(), train_size=0.8)
train['review'] = train['review'].apply(process_review)
test['review'] = test['review'].apply(process_review)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', sgd),
                     ])

# Train model and perform test predictions.
text_clf = text_clf.fit(train['review'], train['fake'])
predicted = text_clf.predict(test['review'])

# Evaluate test predictions.
print '\nTest Data Performance:\n'
print metrics.classification_report(test['fake'], predicted)

# Predict whether Yelp hotel reviews are fake or not.
reviews = pd.read_csv(HOTEL_REVIEWS)
predicted_reviews = text_clf.predict(reviews['text'])

print '\nTotal Reviews: ' + str(len(predicted_reviews)) + ", Fake: " + str(np.ndarray.tolist(predicted_reviews).count(1)) + '\n'

print 'Real Reviews:\n'
for review, prediction in itertools.izip(reviews.iterrows(), predicted_reviews):
    if prediction == 0:
        print str(review[1]['business_name']) + ": Review ID - " + str(review[1]['review_id']) + ", Review - " + review[1]['text']

print '\n Fake Reviews:\n'
for review, prediction in itertools.izip(reviews.iterrows(), predicted_reviews):
    if prediction == 1:
        print str(review[1]['business_name']) + ": Review ID - " + str(review[1]['review_id']) + ", Review - " + review[1]['text']