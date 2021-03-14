---
layout: post
title: Regression Model
subtitle: Sentiment Polarity and BTC Return
cover-img: /assets/img/path3.jpg
thumbnail-img: /assets/img/thumb3.png
share-img: /assets/img/path3.jpg
tags: [Regression]
---

Sentiment Polarity and BTC Return

1: Sentiment Polarity and BTC Return we used
In this part, we use TextBlob to get sentiment polarity.

2: Sentiment Polarity Regression Model
In this part, we use simple OLS regression model below:

3: Conclusion
From the result above, we can see the sentiment polarity has relatively low relationship with all the returns.

4: Potential Problems
Here are some potential problems in the model.
(1)	The TextBlob may not be able to analyze bloggers text accurately since there may be some financial words and discussions related to BTC.
(2)	These Bloggers may not be representative. It’s hard to choose suitable bloggers to represent the whole point towards BTC in Twitter and Reddit
(3)	Though the result in different period seems to show some relationship, but the period may be too short to support this result.

```javascript
import pandas as pd
cleaned_tweets = pd.read_csv('cleaned_tweets.csv')
cleaned_tweets.dt = pd.to_datetime(cleaned_tweets.dt)
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
