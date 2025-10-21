# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('part7_natural_language_processing/Restaurant_Review.tsv', delimiter = '\t',
                      quoting = 3) # as we have quoting text (like: "yum yum sauce"), we set quoting parameter here to 3, which means "no quoting", to perevent processing errors

# Cleaning the texts
import re # Regular expression operations library of Python, it allows us to deal with special characters
import nltk # The Natural Language Toolkit (NLTK) is a Python package for natural language processing. We need to install it first: pip install nltk



# Creating the Bag of Words model


# Splitting the dataset into the Training set and Test set


# Training the Naive Bayes model on the Training set


# Predicting the Test set results


# Making the Confusion Matrix