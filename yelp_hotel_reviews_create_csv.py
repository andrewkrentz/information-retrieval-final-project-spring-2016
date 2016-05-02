"""
Andrew Krentz
Information Retrieval
Spring 2016

Final Project
Yelp Data Analysis
yelp_hotel_reviews_create_csv.py

Converts Yelp business data to CSV format and selects businesses that include
'Hotels' or 'Hotels & Travel' in their categories.
"""

import json
import unicodecsv as csv

INPUT_FILE = 'data/yelp_academic_dataset_review.json'
BUSINESS_INPUT_FILE = 'data/yelp_academic_dataset_business.json'
OUTPUT_FILE = 'data/yelp_hotel_reviews.csv'

business_input = open(BUSINESS_INPUT_FILE, 'r')
input = open(INPUT_FILE, 'r')
output = open(OUTPUT_FILE, 'wb')

hotels = dict()

for line in business_input:
    business = json.loads(line)
    if 'Hotels' in business['categories'] or 'Hotels & Travel' in business['categories']:
        hotels[business['business_id']] = business['name']

hotel_reviews = []

for line in input:

    if len(hotel_reviews) > 1000:
        break

    review = json.loads(line)
    if review['business_id'] in hotels.keys():
        data = dict()
        data['review_id'] = review['review_id']
        data['business_id'] = review['business_id']
        data['business_name'] = hotels[data['business_id']]
        data['stars'] = review['stars']
        data['date'] = review['date']
        data['text'] = review['text'].replace('\n', ' ')
        hotel_reviews.append(data)

csv_writer = csv.DictWriter(output, ['business_id', 'business_name', 'review_id', 'date', 'stars', 'text'])
csv_writer.writeheader()
csv_writer.writerows(hotel_reviews)
