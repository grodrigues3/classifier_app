# classifier_app
Author: Garrett Rodrigues

-------------------------------------------------------------------------
This is a web application written using python and flask that satisfies the following:
"""
Create a web application that does the following:
- On the client side, create a web interface to enter a labeled text dataset in any format you want (or multiple formats).
- Implement a machine learning algorithm (at least logistic regression, but feel free to explore other options for better performance) on the server side to train a classifier with this dataset.
- When the classifier is trained, show results on the web page and let the user test it on new data.

Implementation details:
- You can use any language on the server side, and JavaScript on the client side.
- You should not use existing packages with off-the-shelf logistic regression implementations like scikit-learn.
- You will find attached one of our example datasets to help you getting started.
"""

Details:

The front end uses the flask micro framework to build a server side web application.
The back end models are built with numpy and scipy and allow one to train a logistic regression model
  using either the scipy optimization libraries or stochastic gradient descent.
