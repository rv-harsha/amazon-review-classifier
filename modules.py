from __future__ import print_function
from six.moves import cPickle as pickle

import os
import sys
import nltk
import gzip
import json
import string
import urllib
import calendar
import numpy as np
import pandas as pd
import seaborn as sns
sns.set()

from time import time
import matplotlib.pyplot as plt
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')

nltk.download('stopwords')
stopwords = set(nltk.corpus.stopwords.words('english'))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem.snowball import SnowballStemmer

DATASET_NAME = 'reviews_Kindle_Store.json.gz'
FEATURES_PKL_FILENAME = 'features.pkl'
LABELS_PKL_FILENAME = 'labels.pkl'
pickle_file_name = (DATASET_NAME[:-8]+'.pkl')
DATASET_SIZE = 725133804
MIN_HELPFULNESS_REVIEWS = 8
THRESHOLD = 0.5
FIG_SIZE = (14, 8)
RAN_STATE = 42  # Random state for classifiers

cwd = os.getcwd()

url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/'
last_percent_reported = None

# This code is borrowed from https://jmcauley.ucsd.edu/data/amazon/
def download_progress_hook(count, blockSize, totalSize):
    """A hook to report the progress of a download. This is mostly intended for users with
    slow internet connections. Reports every 1% change in download progress.
    """
    global last_percent_reported
    percent = int(count * blockSize * 100 / totalSize)

    if last_percent_reported != percent:
        if percent % 5 == 0:
            sys.stdout.write("%s%%" % percent)
            sys.stdout.flush()
        else:
            sys.stdout.write(".")
            sys.stdout.flush()

        last_percent_reported = percent

# This code is borrowed from https://jmcauley.ucsd.edu/data/amazon/
def maybe_download(filename, expected_bytes, force=False):
    """Download a file if not present, and make sure it's the right size."""
    if force or not os.path.exists(filename):
        print('Attempting to download:', filename)
        filename, _ = urllib.request.urlretrieve(
            url + filename, filename, reporthook=download_progress_hook)
        print('\nDownload Complete!')
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified', filename)
    else:
        raise Exception(
            'Failed to verify ' + filename + '. Can you get to it with a browser?')
    return filename

# This code is borrowed from https://jmcauley.ucsd.edu/data/amazon/
def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

# This code is borrowed from https://jmcauley.ucsd.edu/data/amazon/
def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

# This code is borrowed from https://jmcauley.ucsd.edu/data/amazon/
def extract_and_load():

    if os.path.exists(pickle_file_name):
        print('Pickled file already present, loading...')
        data = pd.read_pickle(pickle_file_name)
        print('Pickle file loaded.')
    else:
        data = getDF(DATASET_NAME)
        print('Creating Pickle file.')
        data.to_pickle(pickle_file_name)
    print('Data loaded to Dataframe')

    return data

# This code is modified and used from https://colab.research.google.com/drive/1Bdc_YOd2I8ne5BUy2PQ6nQJIpI0GeCA2#scrollTo=fGmxjcS50rqj
def exploratory_data_analysis():

    data = pd.read_pickle('eda.pkl')

    f, axes = plt.subplots(2,2, figsize=(12,8))
    #--- Yearly Reviews
    yearly = data.groupby(['year'])['reviewerID'].count().reset_index()
    yearly = yearly.rename(columns={'reviewerID':'no_of_reviews'})
    yearChart = sns.lineplot(x='year',y='no_of_reviews',data=yearly, ax = axes[0,0])
    yearChart.set_title('No of reviews over years')

    #--- Monthly Reviews
    monthly = data.groupby(['month'])['reviewerID'].count().reset_index()
    monthly['month'] = monthly['month'].apply(lambda x : calendar.month_name[x])
    monthly = monthly.rename(columns={'reviewerID':'no_of_reviews'})
    monthChart = sns.barplot(x='month',y='no_of_reviews',data=monthly, ax = axes[0,1])
    monthChart.set_title('No of reviews over month')
    monthChart.set_xticklabels(monthChart.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    #-- Getting overall ratings for every kindle products
    sns.countplot(x = 'overall', data = data, ax = axes[1,0] ).set_title('Overall Reviews')

    #--- helpfulness of review.
    helpfulness = data[['helpful','asin']]
    helpfulness[['helpfulVotes','totalVotes']] = pd.DataFrame(helpfulness.helpful.values.tolist(), index=helpfulness.index)
    helpfulness = helpfulness.drop(['helpful'], axis = 1)
    #--- calculating helpfulness Percentage
    helpfulness['percentage'] = (helpfulness.helpfulVotes/helpfulness.totalVotes)*100
    helpfulness = helpfulness.fillna(0)
    final_helpfullness = helpfulness.groupby(pd.cut(helpfulness.percentage,np.arange(0,101,10))).count()
    final_helpfullness = final_helpfullness.rename(columns={'percentage':'count'})
    final_helpfullness = final_helpfullness.reset_index()
    helpfullnessChart = sns.barplot(x='percentage',y='count',data=final_helpfullness, ax = axes[1,1])
    helpfullnessChart.set_title('helpfullness of reviews ranked by percentage')
    helpfullnessChart.set_xticklabels(helpfullnessChart.get_xticklabels(), rotation = 45, horizontalalignment = 'right')

    f.tight_layout()
    plt.show()

    #--- mean median and mode of overall ratings
    f = plt.figure(figsize=(18,10))
    #---mean
    stat_reviews_yearly = data.groupby(['year'])['overall'].mean().reset_index()
    stat_reviews_yearly = stat_reviews_yearly.rename(columns={'overall':'mean_overall'})
    #---median
    median_yearly = data.groupby(['year'])['overall'].median().reset_index()
    stat_reviews_yearly['median_overall'] = median_yearly['overall']
    #--- plotting the values
    sns.lineplot(x='year',y='mean_overall',data=stat_reviews_yearly, label = 'Mean')
    sns.lineplot(x='year',y='median_overall',data=stat_reviews_yearly, label = 'Median')
    f.tight_layout()
    plt.show()

    # #---mode
    print(data.groupby(['year'])['overall'].value_counts())

    #--- Distribution of number of reviews written by each user
    f = plt.figure(figsize=(18,10))
    userReviews = data[['reviewerID','asin']]
    userReviews = userReviews.groupby(['reviewerID']).count().reset_index()
    userReviews = userReviews.sort_values('asin',ascending = False)
    userReviews = userReviews.rename(columns={'asin':'no of reviews'})
    print(userReviews.head())

    userReviews1 = userReviews.groupby('no of reviews')['reviewerID'].count().reset_index()
    userReviews1 = userReviews1.rename(columns={'reviewerID':'count'})
    print()
    print(userReviews1.head())
    userReviewChart = sns.barplot(x = 'no of reviews',y = 'count',data = userReviews1)
    userReviewChart.set_title('Reviews Written by each user')
    userReviewChart.set_xticklabels(userReviewChart.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    f.tight_layout()
    plt.show()

    f = plt.figure(figsize=(12,10))
    userReviews2 = userReviews1.groupby(pd.cut(userReviews1['no of reviews'],np.arange(0,200,10))).sum()
    userReviews2 = userReviews2.rename(columns={'no of reviews':'range of reveiws'})
    userReviews2 = userReviews2.reset_index()
    userReviewChart = sns.barplot(x='no of reviews',y='count',data=userReviews2)
    userReviewChart.set_title('Reviews Written by each user in a range')
    userReviewChart.set_xticklabels(userReviewChart.get_xticklabels(), rotation = 45, horizontalalignment = 'right')
    f.tight_layout()
    plt.show()

def pre_process_data(data):

    #select the columns
    data = data.iloc[:, [2,3,4,5]]

    #split numerator and denominator
    data['helpful_ratings'] = data['helpful'].apply(lambda x: x[0])
    data['total_ratings'] = data['helpful'].apply(lambda x: x[1])

    # delete un-needed 'helpful catagoryb
    del data['helpful']

    #Check if we have any null values
    print(data.isnull().sum())

    #include reviews that have more than 8 helpfulness data point only
    data = data[(data.total_ratings > MIN_HELPFULNESS_REVIEWS)].copy()

    #transform Helpfulness into a binary variable with threshold ratio
    data.loc[:, 'Helpful'] = np.where(data.loc[:, 'helpful_ratings'] / data.loc[:, 'total_ratings'] > THRESHOLD, 1, 0)

    #Check the balance
    print ('Count:')
    display(data.groupby('Helpful').count())

    if not os.path.exists(LABELS_PKL_FILENAME):
        pickle.dump(data['Helpful'], open(LABELS_PKL_FILENAME, "wb"))

    #Visualize correlation of the data
    correlations = data.corr()
    plt.figure(figsize = FIG_SIZE)
    plt.title("Heatmap of correlations in each catagory")
    _ = sns.heatmap(correlations, vmin=0, vmax=1, annot=True)
    plt.show()

    # Make all text lowercase
    data.reviewText = data.reviewText.apply(lambda x:x.lower())
    data.summary = data.summary.apply(lambda x:x.lower())

    # Remove punctuations
    data.reviewText = data.reviewText.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))
    data.summary = data.summary.apply(lambda x: x.translate(str.maketrans('', '', string.punctuation)))

    # Remove stop words
    data.reviewText = data.reviewText.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))
    data.summary = data.summary.apply(lambda x: " ".join(x for x in x.split() if x not in stopwords))

    # Concat reviewtext and summary data
    data["reviewText"] = data["reviewText"] + data["summary"]

    return data

def extract_features(data):

    #create a stemmer
    stemmer = SnowballStemmer("english")

    #define our own tokenizing function that we will pass into the TFIDFVectorizer. We will also stem the words here.
    def tokens(x):
        x = x.split()
        stems = []
        [stems.append(stemmer.stem(word)) for word in x]
        return stems

    #loads pickle if exists, extracts and pickles if it doesn't
    if os.path.exists(FEATURES_PKL_FILENAME):
        print ('Pickled file already present, loading...')
        features = pickle.load(open(FEATURES_PKL_FILENAME, "rb" ))
        print ('Pickle file loaded.')
    else:
        #define the vectorizer
        vectorizer = TfidfVectorizer(tokenizer = tokens, stop_words = 'english', ngram_range=(1, 1), min_df = 0.01)
        #fit the vectorizers to the data.
        features = vectorizer.fit_transform(data['reviewText'])

        pickle.dump(features, open(FEATURES_PKL_FILENAME, "wb"))

def create_datasets():

    X = pickle.load(open(FEATURES_PKL_FILENAME, "rb"))
    y = pickle.load(open(LABELS_PKL_FILENAME, "rb"))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RAN_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12, random_state=RAN_STATE, stratify=y_train) 

    if not os.path.exists('X.pickle'):
        with open('X.pickle', 'wb') as output:
            pickle.dump([X_train, X_val, X_test], output)

    if not os.path.exists('y.pickle'):
        with open('y.pickle', 'wb') as output:
            pickle.dump([y_train, y_val, y_test], output)

def load_datasets():

    with open('X.pickle', 'rb') as f:
        X_train, X_val, X_test = pickle.load(f)
    with open('y.pickle', 'rb') as f:
        y_train, y_val, y_test = pickle.load(f)

    return X_train, X_val, X_test, y_train, y_val, y_test