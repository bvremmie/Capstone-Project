# IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

#preprocessing
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

# pipelines
from sklearn.pipeline import Pipeline

# NLP transformers
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# classifiers you can use
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


# model selection bits
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score, ShuffleSplit
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold, KFold

# evaluation
from sklearn.metrics import f1_score, accuracy_score

# plotting
from plotting import plot_learning_curve, plot_validation_curve

#UPLOAD DATA
data = pd.read_csv("mbti_1.csv")

#SPLIT DATA INTO LISTS FOR FORUM POSTS AND MBTI TYPES
list_of_posts = data['posts'].to_list()
list_of_types = data['type'].to_list()

#ASSIGN LISTS TO CORRESPONDING VARIABLES
X = list_of_posts
y = list_of_types

functionWords = ["and", 'but', 'the', 'a', 'he', 'him', 'her', 'she', 'they', 'them', 'in', 'under', 'towards', 'before',
                 'of', 'for', 'then', 'well', 'however', 'thus', 'would', 'could', 'should', 'on', 'down', 'this', 'if',
                'also', 'or']

stopWords = ['a', 'and', 'an', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
             'that', 'the', 'to', 'was', 'were', 'will', 'with', 'nor', 'yet', 'so', 'she', 'another', "do", "does", "for",
             "your", "yours", "yourself", "yourselves", "you", "usually", "us", "until", "under", "use", "relate", "quite",
             "none", "not", "to", "towards", "that", "those", "though", "through", "this", "then", "these", "thing", "their",
             'other', 'of', 'often', 'after', 'any', 'anything', 'anybody', 'anyone', 'anyhow', 'anywhere', 'another', 'around',
             'again', 'are', 'above', 'about', 'already', 'always', 'also', 'although', 'almost', 'all', 'very', 'get', 'both',
             'getting', 'by', 'both', 'be', 'became', 'becomes', 'become', 'behind', 'between', 'beneath', 'been', 'below',
             'besides', 'beside', 'much', 'must', 'meanwhile', 'mostly', 'most', 'moreover', 'more', 'he', 'hers', 'herself',
             'her', 'had', 'having', 'have', 'has', 'him', 'himself', 'his', 'however', 'how', 'would', 'was', 'with', 'which',
             'whichever', 'while', 'said', 'seem', 'seems', 'in', 'from', 'for', 'did', 'even']

# REMOVING STOP WORDS
for i in range(len(X)):
    for j in range(len(stopWords)):
        X[i] = X[i].replace(' '+stopWords[j]+' ', ' ')
        X[i] = X[i].replace(' '+stopWords[j].capitalize()+' ', ' ')

# REMOVING FUNCTION WORDS
for i in range(len(X)):
    for j in range(len(functionWords)):
        X[i] = X[i].replace(' '+functionWords[j]+' ', ' ')
        X[i] = X[i].replace(' '+functionWords[j].capitalize()+' ', ' ')

# REMOVING Y PERSONAL TYPES FROM POSTS IF THEY EXIST
for i in range(len(list_of_posts)):
    list_of_posts[i] = list_of_posts[i].replace(list_of_types[i], '')

print(list_of_posts[1])

#SPLIT DATA AND VECTORIZE
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.9, stratify = y)

count_vect = CountVectorizer()
count_vect.fit(X_train)
X_train_vect = count_vect.transform(X_train)
X_test_vect = count_vect.transform(X_test)