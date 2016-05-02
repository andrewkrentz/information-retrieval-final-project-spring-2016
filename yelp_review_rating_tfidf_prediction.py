"""
Andrew Krentz
Information Retrieval
Spring 2016

Final Project
Yelp Data Analysis
yelp_reviews_rating_tfidf_prediction.py

Converts the Yelp reviews JSON format into CSV format.
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
from sklearn.metrics import mean_squared_error

REVIEWS_FILE = 'data/yelp_reviews.csv'

# Downloads the NLTK work tokenizer model.
nltk.download('punkt')

# Collection of stop words to filter out.
stop_words = stopwords.words('english')

# Map of punctuation characters to be removed from documents during processing.
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def process_review(review):
    # Remove all punctuation and lowercase the string.
    value_without_punctuation = review.translate(string.maketrans("", ""), string.punctuation).lower()

    # Use NLTK word tokenizer to tokenize the string.
    tokens = [x for x in nltk.word_tokenize(value_without_punctuation) if x not in stop_words]

    # Combine all of the tokens back into a single string and add to results.
    return " ".join(tokens)


df = pd.read_csv(REVIEWS_FILE)

train, test = train_test_split(df.copy(), train_size=0.8)
train['text'] = train['text'].apply(process_review)
test['text'] = test['text'].apply(process_review)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier()),
                     ])

text_clf = text_clf.fit(train['text'], train['stars'])
predicted = text_clf.predict(test['text'])

print '\nMean Squared Error: ' + str(mean_squared_error(test['stars'], predicted))
