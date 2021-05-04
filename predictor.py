# Assuming draws.csv already exists.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor, ClassifierChain, RegressorChain
from sklearn import metrics
from datetime import date, datetime
import matplotlib.pyplot as plt

df = pd.read_csv('draws.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
df['Date'] = (df['Date'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')

X = np.array(df[['Draw Number', 'Date', 'Jackpot']])
y = np.array(df[['1','2','3','4','5','PB']])

x_train, x_test, y_train, y_test = train_test_split(X,y)

classifier = MultiOutputClassifier(RandomForestClassifier(n_estimators=36), n_jobs=-1)
classifier.fit(x_train, y_train)
print('Chance for winning: ', classifier.score(x_test, y_test))
