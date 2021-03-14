---
layout: post
title: Introduction and Data Preprocessing
subtitle: Natural Language Processing Group Project
cover-img: /assets/img/path1.jpg
thumbnail-img: /assets/img/thumb1.png
share-img: /assets/img/path1.jpg
tags: [Introduction, Data Preprocessing]
---

Introduction and Data Preprocessing

Background:

The Internet has grown tremendously in the past decade and with the invention of social media like Twitter, Facebook,  sharing knowledge and experiences has become easy. For example, hundreds of thousands of Twitter users generate huge volumes of tweets data every day related to Bitcoin. 

Bitcoin is a decentralized electronic currency system. It has gathered a lot of attraction from people during the recent times and its price has been volatile, which just fluctuates constantly on real time like a stock exchange. 

But there is no efficient way of predicting the price of Bitcoin Price even though we dig deeper into the Blockchain. Given that We know that people express their opinions and sentiments through online portals, So it is very important to build a model that can predict the price of Bitcoin using the social media data from the internet. Twitter and Rebbit are perfect social media platforms for us.

Model Used:

In the project, we use the three types of Sentiment analysis Classifiers, the simple OLS regression model，and some popular methods of Machine Learning to do sentiment regression analysis and check our results.

Our goal is to capture the Bitcoin return from the sentiment analysis and to predict price  fluctuation for guiding investors’ practices.

Get Text Data:

First, we scrape bitcoin price data and bloggers’ tweets to get the original tweets data. We also use PRAW package to scrape Reddit blogs.

Clean Text Data:

And in this part we use re package to remove some unimportant information in text, and save the cleaned data.

```javascript
# Get Twitter Text Data

import pandas as pd
import twint
import nest_asyncio
nest_asyncio.apply()

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

```javascript
# Clean Text Data

import re

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
