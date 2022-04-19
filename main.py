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

# CREATE LISTS FOR OCCURENCE TRAIT IN EACH PERSONALITY TYPE

y_train_ie = []
y_train_sn = []
y_train_tf = []
y_train_jp = []

for label in y_train:
    if "I" in label:
        y_train_ie.append(0)
    if "E" in label:
        y_train_ie.append(1)
    if "S" in label:
        y_train_sn.append(0)
    if "N" in label:
        y_train_sn.append(1)
    if "T" in label:
        y_train_tf.append(0)
    if "F" in label:
        y_train_tf.append(1)
    if "J" in label:
        y_train_jp.append(0)
    if "P" in label:
        y_train_jp.append(1)

#print(y_train_ie[:10])
#print(y_train[:10])

#CREATE FUNCTION TO MAKE SURE THERES A EVEN BALANCE OF LETTERS
def even_traits_amt(y_train_traits, X_train):
    first_trait_count = 0
    second_trait_count = 0
    X_train_traits_adj = []
    y_train_traits_adj = []
    first_trait_greater = False

    if y_train_traits.count(0) > y_train_traits.count(1):
        first_trait_greater = True
        first_trait_limit = y_train_traits.count(1)
    else:
        second_trait_limit = y_train_traits.count(0)


    for i in range(len(y_train_traits)):
        if first_trait_greater:
            if y_train_traits[i] == 1:
                X_train_traits_adj.append(X_train[i])
                y_train_traits_adj.append(y_train_traits[i])
            else:
                if first_trait_count < first_trait_limit:
                    X_train_traits_adj.append(X_train[i])
                    y_train_traits_adj.append(y_train_traits[i])
                    first_trait_count += 1
        else:
            if y_train_traits[i] == 0:
                X_train_traits_adj.append(X_train[i])
                y_train_traits_adj.append(y_train_traits[i])
            else:
                if second_trait_count < second_trait_limit:
                    X_train_traits_adj.append(X_train[i])
                    y_train_traits_adj.append(y_train_traits[i])
                    second_trait_count += 1

    return X_train_traits_adj, y_train_traits_adj


X_train_ie_adj, y_train_ie_adj = even_traits_amt(y_train_ie, X_train)
print(f"I:{y_train_ie_adj.count(0)}, E:{y_train_ie_adj.count(1)}")
X_train_sn_adj, y_train_sn_adj = even_traits_amt(y_train_sn, X_train)
print(f"S:{y_train_sn_adj.count(0)}, N:{y_train_sn_adj.count(1)}")
X_train_tf_adj, y_train_tf_adj = even_traits_amt(y_train_tf, X_train)
print(f"T:{y_train_tf_adj.count(0)}, F:{y_train_tf_adj.count(1)}")
X_train_jp_adj, y_train_jp_adj = even_traits_amt(y_train_jp, X_train)
print(f"J:{y_train_jp_adj.count(0)}, P:{y_train_jp_adj.count(1)}")

# I/E ADJ ADA SCORES 1-3
my_decision_tree = RandomForestClassifier(max_depth=10, min_samples_leaf=20)
aclf = AdaBoostClassifier(my_decision_tree, n_estimators=100)

dt_pipe = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),('ie_aclf', aclf)])
scores = cross_val_score(dt_pipe, X_train_ie_adj, y_train_ie_adj, cv=5, scoring = 'accuracy')
print(scores.mean())
print(scores)

# S/N ADJ ADA SCORES 1-3
my_decision_tree = RandomForestClassifier(max_depth=10, min_samples_leaf=20)
aclf = AdaBoostClassifier(my_decision_tree, n_estimators=100)

dt_pipe = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),('sn_aclf', aclf)])
scores = cross_val_score(dt_pipe, X_train_sn_adj, y_train_sn_adj, cv=5, scoring = 'accuracy')
print(scores.mean())
print(scores)

# T/F ADJ ADA SCORES 1-3

my_decision_tree = RandomForestClassifier(max_depth=10, min_samples_leaf=20)
aclf = AdaBoostClassifier(my_decision_tree, n_estimators=100)

dt_pipe = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),('tf_aclf', aclf)])
scores = cross_val_score(dt_pipe, X_train_tf_adj, y_train_tf_adj, cv=5, scoring = 'accuracy')
print(scores.mean())
print(scores)

#J/P ADJ ADA SCORES 1-3

my_decision_tree = RandomForestClassifier(max_depth=10, min_samples_leaf=20)
aclf = AdaBoostClassifier(my_decision_tree, n_estimators=100)

dt_pipe = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),('jp_aclf', aclf)])
scores = cross_val_score(dt_pipe, X_train_jp_adj, y_train_jp_adj, cv=5, scoring = 'accuracy')
print(scores.mean())
print(scores)