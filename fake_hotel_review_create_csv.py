"""
Andrew Krentz
Information Retrieval
Spring 2016

Final Project
Yelp Data Analysis
spam_create_csv.py

Converts the fake review data into CSV format. The data was collected by Myle Ott for
the following paper and is located at http://myleott.com/op_spam/:

[1] M. Ott, C. Cardie, and J.T. Hancock. 2013. Negative Deceptive Opinion Spam.
In Proceedings of the 2013 Conference of the North American Chapter of the
Association for Computational Linguistics: Human Language Technologies.
"""

import csv
import os

DATA_DIR = 'data/op_spam_v1.4'
OUTPUT_FILE = 'data/op_spam.csv'

output = open(OUTPUT_FILE, 'wb')
spam_data = []

for root, subdirs, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".txt"):
            record = dict()
            record['filename'] = root + '/' + file
            record['fake'] = 1 if file.startswith('t') else 0
            record['positive'] = 1 if 'positive' in root else 0

            with open(record['filename'], 'r') as f:
                record['review'] = f.read().replace('\n', ' ')

            spam_data.append(record)

csv_writer = csv.DictWriter(output, ['fake', 'positive', 'filename', 'review'])
csv_writer.writeheader()
csv_writer.writerows(spam_data)
