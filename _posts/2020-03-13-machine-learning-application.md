---
layout: post
title: Sentiment Analysis
subtitle: Using Twitter & Reddit Text Data
cover-img: /assets/img/path2.jpg
thumbnail-img: /assets/img/thumb2.png
share-img: /assets/img/path2.jpg
tags: [Sentiment]
---

Sentiment Analysis

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
