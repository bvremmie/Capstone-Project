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