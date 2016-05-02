"""
Andrew Krentz
Information Retrieval
Spring 2016

Final Project
Yelp Data Analysis
yelp_restaurants_convert_to_csv.py

Converts the Yelp business JSON data into CSV format and filters out
any businesses that do not have the 'Restaurant' category.
"""

import json
import unicodecsv as csv

INPUT_FILE = 'data/yelp_academic_dataset_business.json'
OUTPUT_FILE = 'data/yelp_restaurants.csv'

input = open(INPUT_FILE, 'r')
output = open(OUTPUT_FILE, 'wb')

restaurant_data = []

for line in input:
    business = json.loads(line)
    if 'Restaurants' in business['categories']:
        restaurant = dict()
        restaurant['stars'] = business['stars']
        restaurant['business_id'] = business['business_id']
        restaurant['name'] = business['name']
        restaurant['review_count'] = business['review_count']
        restaurant['number_of_days_per_week_open'] = len(business['hours'].keys())
        restaurant['location'] = business['city'] + '  ' + business['state']
        restaurant['Alcohol'] = business['attributes'].get('Alcohol', 'unknown')
        restaurant['Attire'] = business['attributes'].get('Attire', 'unknown')
        restaurant['Noise Level'] = business['attributes'].get('Noise Level', 'unknown')
        restaurant['Takes Reservations'] = business['attributes'].get('Takes Reservations', 'unknown')
        restaurant['Accepts Credit Cards'] = business['attributes'].get('Accepts Credit Cards', 'unknown')
        restaurant['Price Range'] = business['attributes'].get('Price Range', 'unknown')
        restaurant['Fast Food'] = 'Fast Food' in business['categories']
        restaurant['Japanese'] = 'Japanese' in business['categories']
        restaurant['American (Traditional)'] = 'American (Traditional)' in business['categories']

        if 'Ambience' in business['attributes']:
            restaurant['romantic'] = business['attributes']['Ambience'].get('romantic', 'unknown')
            restaurant['intimate'] = business['attributes']['Ambience'].get('intimate', 'unknown')
            restaurant['hipster'] = business['attributes']['Ambience'].get('hipster', 'unknown')
            restaurant['upscale'] = business['attributes']['Ambience'].get('upscale', 'unknown')
        else:
            restaurant['romantic'] = 'unknown'
            restaurant['intimate'] = 'unknown'
            restaurant['hipster'] = 'unknown'
            restaurant['upscale'] = 'unknown'

        restaurant_data.append(restaurant)

csv_writer = csv.DictWriter(output, restaurant_data[0].keys())
csv_writer.writeheader()
csv_writer.writerows(restaurant_data)

