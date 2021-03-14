---
layout: post
title: Machine Learning Application
subtitle: In the Prediction of Positive or Negative Return of Bitcoin
cover-img: /assets/img/path4.jpg
thumbnail-img: /assets/img/thumb4.png
share-img: /assets/img/path4.jpg
tags: [Machine Learning]
---

In the prediction of the positive or negative return of Bitcoin

In this part, we use the machining learning to explore and predict the negative or positive return of the incoming single day; and the return of holding Bitcoin for several accumulated days.

As the previous analysis, we know that the sentiment classifier we use, either Text Blob, Affin, or the Vader can not perform well to forecast the positive or the negative return of Bitcoin independently.

We want to see whether if we can use all these factors all together to predict. So we introduce the machining learning.
As for the inputs, in the chart, from our text analysis, we can see the seven sentiment features of our data set.

As for the targets, we would like to explore whether the seven sentiment features might be used to predict the accumulated return and the single daily return.

The accumulated return means that the return of holding the bitcoin for several days, and our task is to forecast whether the return is positive or negative; besides we short negative for N and short positive for P.

We used 10-fold cross-validation and test the traditional machining learning model, such as SVM, artificial neutral networks, logistics, naïve Bayes, decision tree, KNN and so on. 

We can see that Logistics and Artificial neural networks performs well in the prediction of accumulated return.

The below is about five days accumulated return by artificial neutral network.  And we can see the percent of correctly classified instances is over 67%.

Here comes our conclusion:
-From the result above, we can see the single sentiment polarity has relatively low relationship with all the returns.
-If we used the three sentiment polarity methods all together, Affin, Vader and Blob text and we used these sentiment score all as the inputs of machining learning method, we could predict the positive or negative return of 4-7 days accumulated return well.
-Besides, the correct rate is above 65%+
