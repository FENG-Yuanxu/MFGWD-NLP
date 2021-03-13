---
layout: post
title: Introduction And Data Preprocessing
subtitle: Natural Language Processing Group Project
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [introduction, data preprocessing]
---

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
