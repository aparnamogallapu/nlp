# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 22:49:24 2019

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv('C:\\Users\\HP\\Downloads\\internship\\emails.csv')
#Text=dataset['text'][0]
dataset["text"]=dataset.text.astype("str").transform(lambda x:x.replace("Subject:",''))
dataset["text"]=dataset.text.astype("str").transform(lambda x:x.replace("https",''))
dataset["text"]=dataset.text.astype("str").transform(lambda x:x.replace("com",''))
dataset["text"]=dataset.text.astype("str").transform(lambda x:x.replace("co",''))

import re
import nltk
nltk.download('stopwords')
#nltk.download('wordnet')
#from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

corpus=[]
for i in range(0,5728):
    Text=re.sub('[^a-zA-Z]',' ',dataset['text'][i])
    Text.lower()
    Text=Text.split()
    Text=[ps.stem(word) for word in Text if not word in set(stopwords.words('english'))]
    Text=' '.join(Text)
    corpus.append(Text)
    
    
    

'''wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(Text)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()'''

##creating bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5700)
x=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values


#spliting the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)


##
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
classifier=LogisticRegression()
classifier.fit(x_train,y_train)
classifier.score(x_test,y_test)

y_pred=classifier.predict(x_test)
y_pred

##confusion matrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

###
from sklearn.model_selection import cross_val_score,KFold
kfold=KFold(n_splits=10)
score=cross_val_score(classifier,x,y,cv=kfold,scoring="accuracy")
score.mean()
score
print('score:',score.mean)
print('score:',score.mean())



