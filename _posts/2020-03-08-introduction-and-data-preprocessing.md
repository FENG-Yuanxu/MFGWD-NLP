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



Data Analysis:

For our first phase of data analysis, we thought knowing what big bloggers feel about Bitcoin is extremely valuable to capture the Bitcoin Price fluctuation. So we first look at how negative or positive felt by big bloggers. We used 3 lexicon-based tweet polarity classifiers to calculate sentiment scores respectively. 

We first use AFFIN, which are scored ranging between -5 to +5. In the twitter sentiment analysis using AFFIN, we would be evaluating the overall average sentiment score for the extracted text. 

Secondly, we use VADER. This works just like the AFFIN as it also has word weights ranging from positive to negative. But VADER is aware of the social media jargon used by twitter users. 

Lastly, we use TextBlob. By feeding the unique tweets, we obtain polarity as the output that ranges between -1 to +1. So, a tweet has Positive sentiment when it’s polarity is greater than 0 and negative sentiment when it’s polarity is lesser than 0. The TextBlob also gives the Subjectivity of a tweet.

After analysis, we aggregated the daily mean sentiment score for our dataset. By calculating the maximum and minimum scores, we choose the threshold for each model. Then we plotted their positive and negative polarity on the graphs. We can roughly observe that most of the tweets talking about Bitcoin are positive. We saved the scores for future use.

After the above steps, we found that Sentiment analysis can be extremely valuable in terms of quantifying the emotional of any given sample of text, but visualizations of sentiment scores can often be underwhelming. For this reason, we decided to dive into a different type of analysis: word frequency. We will use the WordCloud library to display word cloud of the most positive and negative words based on the sentiment scores we get previously.

Here are some observations. As you can see, “positive” frequently contain words like “great”, “legendary”, “join”, and “best”. We also have negative sentiment word cloud for Bitcoin. In this case, it contains words like “hate”, “hell”, and “funeral”

The word clouds served as a great stimulus with which to gather an initial impression of the corpus, but we want to know more about the relationship between polarity score with Bitcoin daily return. Now let’s welcome my teammate to talk more about this part.
