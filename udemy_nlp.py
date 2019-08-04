# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:39:06 2019

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

dataset = pd.read_csv('C:\\Users\\HP\\Desktop\\u_datasets\\Natural_Language_Processing\\Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)
#review=dataset['Review'][0]
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]','',dataset['Review'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
#creating Bag of words:
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

##splitting the data train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


##fitting naive_bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

#predicting the test set results
y_pred=classifier.predict(x_test)
y_pred

#confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
(55+91)/200

#########
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model_log=LogisticRegression()
model_log.fit(x_train,y_train)
model_log.score(x_test,y_test)

from sklearn.model_selection import cross_val_score,KFold

kfold=KFold(n_splits=10)
score=cross_val_score(model_log,x,y,cv=kfold,scoring="accuracy")
score.mean()
score
print('Score:',score.mean)
print('Score:',score.mean())
