"""
Andrew Krentz
Information Retrieval
Spring 2016

Final Project
Yelp Data Analysis
yelp_reviews_sentiment.py

Uses the NLTK Sentiment Analysis library to try to predict the ratings
of user reviews.

Much of the code is taken from the NLTK documentation example:
http://www.nltk.org/howto/sentiment.html
"""

import string
import pandas as pd
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR

# The file that contains the reviews in CSV format.
REVIEWS_FILE = 'data/yelp_reviews.csv'

# Collection of stop words to filter out.
stop_words = stopwords.words('english')

# Map of punctuation characters to be removed from documents during processing.
remove_punctuation_map = dict((ord(char), None) for char in string.punctuation)


def tokenize_review(review):
    """
    Tokenize the review text.
    """
    # Remove all punctuation and lowercase the string.
    value_without_punctuation = review.translate(string.maketrans("", ""), string.punctuation).lower()

    # Use NLTK word tokenizer to tokenize the string.
    return [x for x in nltk.word_tokenize(value_without_punctuation) if x not in stop_words]


df = pd.read_csv(REVIEWS_FILE)

# Create the train and test sets.
train, test = train_test_split(df.copy(), train_size=0.8)
train['tokenized_text'] = train['text'].apply(tokenize_review)
test['tokenized_text'] = test['text'].apply(tokenize_review)

# The NLTK sentiment classifier has its own special tuple format for its input.
train_tuples = []
for row in train.iterrows():
    train_tuples.append((row[1]['tokenized_text'], row[1]['stars']))

test_tuples = []
for row in test.iterrows():
    test_tuples.append((row[1]['tokenized_text'], row[1]['stars']))

# Extract features from the train sets.
sentim_analyzer = SentimentAnalyzer()
all_words_neg = sentim_analyzer.all_words([mark_negation(text) for text in train_tuples])
unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)

# Transform the train and test set into their features.
training_set = sentim_analyzer.apply_features(train_tuples)
test_set = sentim_analyzer.apply_features(test_tuples)

# Train the classifier.
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

# Use the classifier to predict the rating for each review in the test set.
predictions = []
for feature in test_set:
    predictions.append(classifier.classify(feature[0]))

# Compute the MSE of the predictions.
print '\nMean Squared Error: ' + str(mean_squared_error(test['stars'], predictions))
