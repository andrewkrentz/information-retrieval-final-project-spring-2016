"""
Andrew Krentz
Information Retrieval
Spring 2016

Final Project
Yelp Data Analysis
yelp_restaurants_regressor.py

Attempts to predict the stars of a restaurant based upon its features.
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn_pandas import DataFrameMapper
import sklearn.preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import train_test_split
import itertools

INPUT_FILE = 'data/yelp_restaurants.csv'
PREDICTION_FILE = 'data/yelp_restaurant_predictions.txt'

# Read in data set.
df = pd.read_csv(INPUT_FILE)
df = df[np.isfinite(df['stars'])]
df['stars'] = df['stars'].astype(float)
df['review_count'] = df['review_count'].astype(float)
df['number_of_days_per_week_open'] = df['number_of_days_per_week_open'].astype(float)
df['location'] = df['location'].astype(basestring)

# Split data into train and test sets.
msk = np.random.rand(len(df)) < 0.8
train = df[msk].drop(['business_id', 'name'], axis=1)
test = df[~msk]

# Convert fields into sklearn features.
feature_mapper = DataFrameMapper([
    (['review_count'], sklearn.preprocessing.StandardScaler()),
    (['number_of_days_per_week_open'], sklearn.preprocessing.StandardScaler()),
    (['Price Range'], sklearn.preprocessing.LabelBinarizer()),
    (['Attire'], sklearn.preprocessing.LabelBinarizer()),
    (['Noise Level'], sklearn.preprocessing.LabelBinarizer()),
    (['Alcohol'], sklearn.preprocessing.LabelBinarizer()),
    (['Takes Reservations'], sklearn.preprocessing.LabelBinarizer()),
    (['Accepts Credit Cards'], sklearn.preprocessing.LabelBinarizer())
])

# Create numpy array with training features that sklearn expects as an input.
train_array = np.round(feature_mapper.fit_transform(train.copy().drop('stars', axis=1)), 2)

# Create linear regression object
regr = linear_model.LinearRegression()
model = regr.fit(train_array, train['stars'].values)

# Alternative regression learning algorithm using support vector machines.
# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# model = svr_rbf.fit(train_array, train['stars'].values)

# Predict the star values for the test set.
test_array = np.round(feature_mapper.fit_transform(test.copy().drop(['business_id', 'name', 'stars'], axis=1)), 2)
predictions = np.round(model.predict(test_array), 1)

# If predictions are outside of the 1-5 star range then set them to the min/max allowed values.
too_large_prediction_indices = np.asarray(predictions) > 5.0
predictions[too_large_prediction_indices] = 5.0
too_small_predictions_indices = np.asarray(predictions) < 1.0
predictions[too_small_predictions_indices] = 1.0

# Set the prediction values for the test data set.
df.loc[~msk, 'Prediction'] = predictions.tolist()
test = df[~msk]

# Write out the results.
out = open(PREDICTION_FILE, 'w')
for restaurant in test.iterrows():
    out.write(restaurant[1]['name'] + ", Actual Stars - " + str(restaurant[1]['stars']) + ", Prediction - " + str(restaurant[1]['Prediction']) + '\n')

# Calculate the MSE values.
print 'Mean Squared Error: ' + str(mean_squared_error(test['stars'], test['Prediction']))
