---
layout: post
title: Vader Sentiment Analysis
subtitle: Vader WordCloud and Regression Analysis
cover-img: /assets/img/path3.jpg
thumbnail-img: /assets/img/thumb3.png
share-img: /assets/img/path3.jpg
tags: [Vader, WordCloud, Regression]
comments: true
---

## Vader Sentiment Polarity, WordCloud and OLS regression

In this part, we use **Vader** to get sentiment polarity, and we use simple **OLS regression** model.

## Sentiment Polarity and WordCloud

We first use **TextBlob** to create **WordCloud**. Then, we use **Vader**.

Here is our code:
```javascript
# Read Data
import pandas as pd
import numpy as np
from textblob import TextBlob 
import re
import warnings
import nest_asyncio
import scipy as sp # scientific calculation toolkit
import statsmodels.api as sm # statistical models including regression
import statsmodels.formula.api as smf
import linearmodels as lm # linear models including panel OLS
import matplotlib.pyplot as plt
import math
import seaborn as sns
import requests as rq
from wordcloud import WordCloud
nest_asyncio.apply()
warnings.filterwarnings('ignore')

cleaned_tweets = pd.read_csv('cleaned_tweets.csv')
cleaned_tweets.dt = pd.to_datetime(cleaned_tweets.dt)
```

```javascript
# TextBlob wordcloud both positive and negative

from textblob import TextBlob

def sentiment1(x):
    blob = TextBlob(x)
    score = round(blob.sentiment.polarity,4)  
    return score

cleaned_tweets["Text_Blob_polarity"]  = cleaned_tweets["tweet"].apply(sentiment1)

def sentiment1_(x):
    blob = TextBlob(x)
    score = round(blob.sentiment.subjectivity,5)  
    return score

cleaned_tweets["Text_Blob_subjectivity"]  = cleaned_tweets["tweet"].apply(sentiment1_)

x = cleaned_tweets["Text_Blob_polarity"] > 0.7
data = ""
for i in range(len(x)):
    if x[i] == True:
        data = data + cleaned_tweets.loc[i,"tweet"]

from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS
wordcloud = WordCloud(background_color='white',stopwords=STOPWORDS,max_words=200,max_font_size=40,random_state=369).generate(data)

import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('TextBlob_positive.jpg',dpi=600)

x = cleaned_tweets["Text_Blob_polarity"] < -0.5
data = ""
for i in range(len(x)):
    if x[i] == True:
        data = data + cleaned_tweets.loc[i,"tweet"]

wordcloud = WordCloud(background_color='white',stopwords=STOPWORDS,max_words=200,max_font_size=40,random_state=369).generate(data)

import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('TextBlob_negative.jpg',dpi=600)
```

```javascript
# Word Cloud Vader sentiment

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()

def sentiment2(x):
    score = analyser.polarity_scores(x)
    result = [score["pos"], score["neg"], score["neu"], score["compound"]]
    return result

cleaned_tweets['Sentiment_NLTKVader']  = cleaned_tweets['tweet'].apply(sentiment2)

cleaned_tweets[['Vader_Pos','Vader_Neg', 'Vader_Neu', 'Vader_Compound']] = pd.DataFrame(cleaned_tweets.Sentiment_NLTKVader.values.tolist(), index= cleaned_tweets.index)

x = cleaned_tweets["Vader_Pos"] > 0.5
data = ""
for i in range(len(x)):
    if x[i] == True:
        data = data + cleaned_tweets.loc[i,"tweet"]

from subprocess import check_output
from wordcloud import WordCloud, STOPWORDS

wordcloud = WordCloud(background_color='white',stopwords=STOPWORDS,max_words=200,max_font_size=40,random_state=369).generate(data)

import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('Vader_positive.jpg',dpi=600)

x = cleaned_tweets["Vader_Neg"] > 0.5
data = ""
for i in range(len(x)):
    if x[i] == True:
        data = data + cleaned_tweets.loc[i,"tweet"]

wordcloud = WordCloud(background_color='white',stopwords=STOPWORDS,max_words=200,max_font_size=40,random_state=369).generate(data)

import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.savefig('Vader_negative.jpg',dpi=600)

cleaned_tweets.to_csv('cleaned_tweets_vader.csv')
```
### OLS regression

Here is our code:
```javascript
btc_prc = pd.read_csv('BTC-USD.csv')
btc_prc['Date'] = pd.to_datetime(btc_prc['Date'])
btc_prc['ret'] = btc_prc['Adj Close'].pct_change()
btc_prc['ret_next'] = btc_prc['ret'].shift(-1)
btc_prc['ret_next2'] = btc_prc['ret'].shift(-2)
btc_prc['ret_next3'] = btc_prc['ret'].shift(-3)
btc_prc['ret_next4'] = btc_prc['ret'].shift(-4)
btc_prc['ret_next5'] = btc_prc['ret'].shift(-5)
btc_prc['ret2'] = btc_prc['Adj Close'].pct_change(2).shift(-2)
btc_prc['ret3'] = btc_prc['Adj Close'].pct_change(3).shift(-3)
btc_prc['ret4'] = btc_prc['Adj Close'].pct_change(4).shift(-4)
btc_prc['ret5'] = btc_prc['Adj Close'].pct_change(5).shift(-5)

fig, ax = plt.subplots(figsize=(12,8))
btc_prc.set_index('Date')['Adj Close'].plot(ax=ax)
ax.set_ylabel('Price')
ax.set_title('BTC Price')
plt.savefig('BTC_price.jpg',dpi=600)

sentiment_score = cleaned_tweets.groupby('dt').mean()
sentiment_score.index.name = 'Date'
sentiment_score.reset_index(inplace=True)

fig, ax = plt.subplots(figsize=(12,8))
sentiment_score.set_index('Date')[['Vader_Pos','Vader_Neg','Vader_Neu','Vader_Compound']].plot(ax=ax)
ax.set_title('Sentiment Score Every Day')
plt.savefig('Sentiment.jpg',dpi=600)

sentiment_score = pd.merge(
    sentiment_score, btc_prc[['Date','ret_next','ret_next2','ret_next3','ret_next4','ret_next5','ret2','ret3','ret4','ret5']],
    on='Date'
)

dic = {}
for ret in ['ret_next','ret_next2','ret_next3','ret_next4','ret_next5','ret2','ret3','ret4','ret5']:
    model = smf.ols('%s ~ Vader_Compound'%ret, data=sentiment_score).fit()
    beta = model.params[1]
    t = model.tvalues[1]
    p = model.pvalues[1]
    r = model.rsquared
    dic[ret] = [beta,t,p,r]
res = pd.DataFrame.from_dict(dic, 'columns')
res.index = ['Polarity','t','p','r']
res.to_csv('res_vader.csv')

smf.ols('ret_next4 ~ Vader_Compound', data=sentiment_score).fit().summary()

fig,ax = plt.subplots(figsize=(14,8))
ax1 = ax.twinx()
sentiment_score.set_index('Date')['Vader_Compound'].plot(ax=ax, label='Vader_Compound(Left)',color='r')
sentiment_score.set_index('Date')['ret_next4'].plot(ax=ax1, label='BTC Daily Return 4 Days Later(Right)',color='b')
ax.set_ylabel('Vader_Compound')
ax1.set_ylabel('Daily Ret')
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax1.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc=0)
plt.savefig('Vader_Compound.jpg',dpi=600)
```

## Conclusion:

From the result above, we can see the sentiment polarity has relatively low relationship with all the returns.

## Potential Problems:

Here are some potential problems in the model.

- The TextBlob may not be able to analyze bloggers text accurately since there may be some financial words and discussions related to BTC.
- These Bloggers may not be representative. It’s hard to choose suitable bloggers to represent the whole point towards BTC in Twitter and Reddit
- Though the result in different period seems to show some relationship, but the period may be too short to support this result.
