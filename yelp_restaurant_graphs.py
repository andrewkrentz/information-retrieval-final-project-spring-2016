"""
Andrew Krentz
Information Retrieval
Spring 2016

Final Project
Yelp Data Analysis
yelp_restaurants_graphs.py

Used for data exploration and graphing some of the Yelp restaurant data.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_FILE = 'data/yelp_restaurants.csv'

df = pd.read_csv(INPUT_FILE)


def histogram_number_of_stars():
    sns.set(style="whitegrid", color_codes=True)
    g = sns.distplot(df['stars'])
    g.axes.set_title('Restaurant Stars Histogram')
    g.set_xlabel("Stars")
    g.set_ylabel("Count")
    fig = g.get_figure()
    fig.savefig("charts/histogram_stars.png")


def boxplot_generic(column):
    sns.set(style="whitegrid", color_codes=True)
    g = sns.boxplot(x=column, y="stars", data=df)
    g.axes.set_title(column + ' Boxplot')
    g.set_xlabel(column)
    g.set_ylabel("Stars")
    fig = g.get_figure()
    fig.savefig("charts/" + str(column) + "_boxplot.png")


def boxplot_number_of_days_per_week_open():
    sns.set(style="whitegrid", color_codes=True)
    g = sns.boxplot(x="number_of_days_per_week_open", y="stars", data=df)
    g.axes.set_title('Number of Days per Week Open Boxplot')
    g.set_xlabel("Days Open per Week")
    g.set_ylabel("Stars")
    fig = g.get_figure()
    fig.savefig("charts/days_per_week_boxplot.png")


def boxplot_attire():
    sns.set(style="whitegrid", color_codes=True)
    g = sns.boxplot(x="Attire", y="stars", data=df)
    g.axes.set_title('Attire Boxplot')
    g.set_xlabel("Attire")
    g.set_ylabel("Stars")
    fig = g.get_figure()
    fig.savefig("charts/attire_boxplot.png")


def boxplot_price_range():
    sns.set(style="whitegrid", color_codes=True)
    g = sns.boxplot(x="Price Range", y="stars", data=df)
    g.axes.set_title('Price Range Boxplot')
    g.set_xlabel("Price Range")
    g.set_ylabel("Stars")
    fig = g.get_figure()
    fig.savefig("charts/price_range_boxplot.png")


def boxplot_romantic():
    sns.set(style="whitegrid", color_codes=True)
    g = sns.boxplot(x="romantic", y="stars", data=df)
    g.axes.set_title('Romantic Boxplot')
    g.set_xlabel("Romantic")
    g.set_ylabel("Stars")
    fig = g.get_figure()
    fig.savefig("charts/romantic_boxplot.png")


def boxplot_hipster():
    sns.set(style="whitegrid", color_codes=True)
    g = sns.boxplot(x="hipster", y="stars", data=df)
    g.axes.set_title('Hipster Boxplot')
    g.set_xlabel("Hipster")
    g.set_ylabel("Stars")
    fig = g.get_figure()
    fig.savefig("charts/hipster_boxplot.png")


def boxplot_location():
    df['location'] = df['location'].astype(basestring)
    sns.set(style="whitegrid", color_codes=True)
    g = sns.boxplot(x="location", y="stars", data=df)
    g.axes.set_title('Location Boxplot')
    g.set_xlabel("Location")
    g.set_ylabel("Stars")
    fig = g.get_figure()
    fig.savefig("charts/location_boxplot.png")


histogram_number_of_stars()
boxplot_generic("Fast Food")
boxplot_generic("Japanese")
boxplot_number_of_days_per_week_open()
boxplot_price_range()
boxplot_attire()
