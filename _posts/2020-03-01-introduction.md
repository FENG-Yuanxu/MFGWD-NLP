---
layout: post
title: Introduction
subtitle: Twint Praw TextBlob and OLS regression
cover-img: /assets/img/path1.jpg
thumbnail-img: /assets/img/thumb1.png
share-img: /assets/img/path1.jpg
tags: [Introduction, Data Preprocessing]
---

## Introduction

We use **Twint** and **Praw** package to scrape data. Then, we use **TextBlob** to analyze sentiment and apply **OLS regression** model.

### Background:

The Internet has grown tremendously in the past decade and with the invention of social media like Twitter, Facebook, sharing knowledge and experiences has become easy. For example, hundreds of thousands of Twitter users generate huge volumes of tweets data every day related to Bitcoin. 

Bitcoin is a decentralized electronic currency system. It has gathered a lot of attraction from people during the recent times and its price has been volatile, which just fluctuates constantly on real time like a stock exchange. 

But there is no efficient way of predicting the price of Bitcoin Price even though we dig deeper into the Blockchain. Given that We know that people express their opinions and sentiments through online portals, So it is very important to build a model that can predict the price of Bitcoin using the social media data from the internet. Twitter and Rebbit are perfect social media platforms for us.

### Model Used:

In the project, we use the three types of **Sentiment analysis Classifiers**, the simple **OLS regression model**，and some popular methods of **Machine Learning** to do sentiment regression analysis and check our results.

Our goal is to capture the Bitcoin return from the sentiment analysis and to predict price  fluctuation for guiding investors’ practices.

### Get Text Data:

First, we scrape bitcoin price data and bloggers’ tweets to get the original tweets data. We also use PRAW package to scrape Reddit blogs.

Here is our code:
```javascript
# Get Twitter Text Data

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

bloggers = ['APompliano','NickSzabo4','nic__carter','CarpeNoctom','Melt_Dem','100trillionUSD','MessariCrypto','TuurDemeester',
           'gavinandresen','NickSzabo4','maxkeiser','rogerkver','CremeDelaCrypto',
            'alexsunnarborg','pmarca','ljxie','jonmatonis','ErikVoorhees']

def financial_blogger_search(blogger, kw, start_date='2020-09-01', end_date='2021-03-01'):
    c = twint.Config()
    c.Search = kw
    c.Username = blogger
    c.Format = "Username: {username} | Date:{date} | Tweet: {tweet}" # Custom output format
    c.Pandas = True
    c.Since = start_date
    c.Until = end_date
    c.Lang = 'en'
    twint.run.Search(c)
    df = twint.storage.panda.Tweets_df
    df.to_csv("%s_%s_%s_%s.csv"%(blogger,kw,start_date,end_date))

for blogger in bloggers:
    financial_blogger_search(blogger,'Bitcoin')

for blogger in bloggers:
    financial_blogger_search(blogger,'BTC')

df = pd.DataFrame()
for kw in ['Bitcoin','BTC']:
    for blogger in bloggers:
        try:
            csv = pd.read_csv("%s_%s_2020-09-01_2021-03-01.csv"%(blogger,kw),index_col=0)[['date','username','tweet']]
            df = pd.concat([df,csv],axis=0)
        except:
            pass

df['dt'] = df.date.apply(lambda x:x[:10])
df.index = range(len(df))
total = df.drop_duplicates().sort_values(['dt','username']).set_index(['dt'])
total.to_csv('total_tweets.csv')

```

### Clean Text Data:

In this part, we use re package to remove some unimportant information in text, and save the cleaned data.

Here is our code:
```javascript
# Clean Text Data

clean = total["tweet"].str.lower()
clean = clean.apply(lambda x :re.sub('@[a-z]*','',x))      # Remove tags
clean = clean.apply(lambda x :re.sub('#[a-z0-9]*','',x))   # Remove hash tags
clean = clean.apply(lambda x :re.sub('[0-9]+[a-z]*',' ',x)) # Remove numnbers and associated text. Like : 1st, 2nd, nth....
clean = clean.apply(lambda x :re.sub('\n','',x))            # Remove \n\t
clean = clean.apply(lambda x :re.sub('https?:\/\/.*',' ',x))        # Remove URLs
clean = clean.apply(lambda x :re.sub('[:;!-.,()%/?|]',' ',x))       # Remove Special characters
clean = clean.apply(lambda x :re.sub('$[a-z]*',' ',x))                        # Remove tickers and strings have $abc pattern
clean = clean.apply(lambda x : x.encode('ascii', 'ignore').decode('ascii'))   # Remove emojis
clean = clean.apply(lambda x :re.sub('[0-9]{4}-[0-9]{2}-[0-9]{2}','',x))      # Remove date
clean = clean.apply(lambda x :re.sub('[0-9]*','',x))
total["tweet"] = clean
total.to_csv('cleaned_tweets.csv')

```

### Sentiment Analysis Using TextBlob Package:

In this part, we use TextBlob package to analyze sentiment.

Here is our code:
```javascript
# BTC Return Calculation And Sentiment Analysis

tc_prc = pd.read_csv('BTC-USD.csv')

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

cleaned_tweets = pd.read_csv('cleaned_tweets.csv')
cleaned_tweets.dt = pd.to_datetime(cleaned_tweets.dt)

cleaned_reddits = pd.read_csv('cleaned_20200901_20210301_tweets_reddits.csv')
cleaned_reddits.columns = ['dt','username','tweet']
cleaned_reddits.dt = pd.to_datetime(cleaned_reddits.dt)

cleaned_tweets = pd.concat([cleaned_tweets, cleaned_reddits]).sort_values('dt')

cleaned_tweets["senti_polarity"] = cleaned_tweets["tweet"].apply(lambda x: TextBlob(x).sentiment.polarity)
cleaned_tweets["senti_subjectivity"] = cleaned_tweets["tweet"].apply(lambda x: TextBlob(x).sentiment.subjectivity)

fig, ax = plt.subplots(figsize=(12,8))
btc_prc.set_index('Date')['Adj Close'].plot(ax=ax)
ax.set_ylabel('Price')
ax.set_title('BTC Price')
plt.savefig('BTC_price.jpg',dpi=600)

sentiment_score = cleaned_tweets.groupby('dt').mean()
sentiment_score.index.name = 'Date'
sentiment_score.reset_index(inplace=True)
sentiment_score

sentiment_score.set_index('Date')[['senti_polarity','senti_subjectivity']].plot(ax=ax)
ax.set_title('Sentiment Polarity & Subjectivity Every Day')
plt.savefig('Sentiment.jpg',dpi=600)

sentiment_score = pd.merge(
    sentiment_score,
    btc_prc[['Date','ret_next','ret_next2','ret_next3','ret_next4','ret_next5','ret2','ret3','ret4','ret5']],
    on='Date'
)

dic = {}
start_date = '2020-09-01'
end_date = '2021-03-01'
for ret in ['ret_next','ret_next2','ret_next3','ret_next4','ret_next5','ret2','ret3','ret4','ret5']:
    model = smf.ols('%s ~ senti_polarity'%ret, 
                    data=sentiment_score[
        (sentiment_score.Date>start_date) & (sentiment_score.Date<end_date)]).fit()
    beta = model.params[1]
    t = model.tvalues[1]
    p = model.pvalues[1]
    r = model.rsquared
    dic[ret] = [beta,t,p,r]
res = pd.DataFrame.from_dict(dic, 'columns')
res.index = ['Polarity','t','p','r']
res.to_csv('res_%s_to_%s.csv'%(start_date,end_date))

smf.ols('ret_next ~ senti_polarity', data=sentiment_score).fit().summary()

fig,ax = plt.subplots(figsize=(14,8))
ax1 = ax.twinx()
sentiment_score.set_index('Date')['senti_polarity'].plot(ax=ax, label='Senti_polarity(Left)',color='r')
sentiment_score.set_index('Date')['ret_next4'].plot(ax=ax1, label='BTC Daily Return 4 Days Later(Right)',color='b')
ax.set_ylabel('Senti_polarity')
ax1.set_ylabel('Daily Ret')
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax1.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc=0)
plt.savefig('polarity_ret.jpg',dpi=600)
```
