import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
import nltk
import re

import string
import collections

#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn import preprocessing

data=pd.read_csv('dataset_00_with_header.csv')

#Remove constant-valued features (x067, x094, x095, x096): no prediction power; 
constant_feat=[]
for i in data.columns:
    a=data[i].nunique()
    if a==1:
        constant_feat.append(i)
print(constant_feat)
        
#Remove 4 features w/ only 1 constant value; 
data.drop(constant_feat, axis=1, inplace=True)

##W/o cross validation; 
#Train-test split; 
xtrain, xtest, ytrain, ytest = train_test_split(data.drop('y', axis=1), data['y'], test_size=0.3, random_state=2018) 

#Imputation;   
medians=xtrain.median()
xtrain.fillna(medians, inplace=True)
xtest.fillna(medians, inplace=True)

#RandomForestRegressor; 
regr = RandomForestRegressor(random_state=1000, n_estimators=200)
regr = regr.fit(xtrain, ytrain)

import pickle
#Save the classifier; 
with open('wh_pred.pickle', 'wb') as f:
    pickle.dump(regr, f)

with open('wh_pred.pickle', 'rb') as f:
    regr=pickle.load(f)
print(regr)

ytest_pred = regr.predict(xtest)

#Performance evaluation;
ytest_preddummy=(abs(ytest_pred-ytest)<=3)
from sklearn.metrics import mean_squared_error, r2_score
RSME=np.sqrt(mean_squared_error(ytest, ytest_pred))
R2=r2_score(ytest, ytest_pred)
Accu=ytest_preddummy.value_counts()/len(ytest_preddummy)

data = [RSME, R2, Accu[True]]
performance_metric = pd.DataFrame(data, index=['RSME', 'R-squared', 'Accuracy'], columns=['Value']) #smaller, bigger, bigger; 
pd.DataFrame(performance_metric).to_csv('performance_metric.txt', sep='\t', header=True, index=True)

print(performance_metric)
pd.DataFrame(ytest_pred).to_csv('y_pred.txt', sep='\t', header=False, index=True)

#Save the median of train dataset; 
medians.to_pickle("trainmedian.pickle")

sns.distplot(ytest-ytest_pred)
