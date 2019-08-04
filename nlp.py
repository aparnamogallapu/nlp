# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:09:09 2019

@author: HP
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset=pd.read_csv("C:\\Users\\HP\\Downloads\\internship\\emails.csv")
dataset.columns
dataset.shape
dataset.info()
dataset["text"]=dataset.text.astype("str").transform(lambda x:x.replace("Subject:",''))
dataset.head()

import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

tk=TweetTokenizer()
ps=PorterStemmer()
lem=WordNetLemmatizer()


def cleaning(s):
    s=str(s)
    s=s.lower()
    s=re.sub('\s\W',' ',s)
    s=re.sub('\W\s',' ',s)
    s=re.sub(r'[^\w]',' ',s)
    s=re.sub('\d+','',s)
    s=re.sub('\s+',' ',s)
    s=re.sub('[!@#$_]','',s)
    s=re.sub('co','',s)
    s=re.sub("https",'',s)
    s=s.replace(',',"")
    s=s.replace("[\w*"," ")
    s=s.lower()
    s=tk.tokenize(s)
    s=[ps.stem(word)for word in s if not word in set(stopwords.words('english'))]
    s=[lem.lemmatize(word) for word in s]
    s=' '.join(s)
    return s
    
dataset["content"]=[cleaning(s)for s in dataset['text']]
dataset['content'][0] 


all_words = ' '.join([text for text in dataset['content']])
from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_words)

plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

normal_words =' '.join([text for text in dataset['content'][dataset['spam'] == 0]])

wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(normal_words)
plt.figure(figsize=(10, 7))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()

def hashtag_extract(x):
    hashtags = []
    for i in x:
        ht =re.findall(r'\w+', i)
        hashtags.append(ht)
    return hashtags

HT_regular = hashtag_extract(dataset['content'][dataset['spam'] == 0])

HT_negative = hashtag_extract(dataset['content'][dataset['spam'] == 1])

HT_regular = sum(HT_regular,[])
HT_negative = sum(HT_negative,[])

import seaborn as sns
a = nltk.FreqDist(HT_regular)
d = pd.DataFrame({'Hashtag': list(a.keys()),
                  'Count': list(a.values())})
# selecting top 10 most frequent hashtags     
d = d.nlargest(columns="Count", n = 10) 
plt.figure(figsize=(16,5))
ax = sns.barplot(data=d, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

b = nltk.FreqDist(HT_negative)
e = pd.DataFrame({'Hashtag': list(b.keys()), 'Count': list(b.values())})
# selecting top 10 most frequent hashtags
e = e.nlargest(columns="Count", n = 10)   
plt.figure(figsize=(16,5))
ax = sns.barplot(data=e, x= "Hashtag", y = "Count")
ax.set(ylabel = 'Count')
plt.show()

from sklearn.feature_extraction.text import  CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1,1,),stop_words=stopwords.words('english')).fit(dataset['content'])
X=vectorizer.transform(dataset['content']).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
train=TfidfTransformer().fit(X)
X=train.transform(X).toarray()

from sklearn.model_selection import train_test_split
train_x,test_x,train_y,text_y=train_test_split(X,dataset.spam.values,test_size=0.25,random_state=0)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model_log=LogisticRegression()
model_log.fit(train_x,train_y)
model_log.score(test_x,text_y)

from sklearn.model_selection import cross_val_score,KFold

kfold=KFold(n_splits=10)
score=cross_val_score(model_log,X,dataset.spam.values,cv=kfold,scoring="accuracy")
score.mean()
score
print('Score:',score.mean)
print('Score:',score.mean())

