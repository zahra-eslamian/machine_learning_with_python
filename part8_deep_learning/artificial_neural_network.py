# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Part 1 - Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values # eliminating the first 3 columns which don't help in model decision
y = dataset.iloc[:, -1].values

# Encoding categorical data

# Label Encoding the "Gender" column

# One Hot Encoding the "Geography" column

# Splitting the dataset into the Training set and Test set

# Feature Scaling


# Part 2 - Building the ANN
# Initializing the ANN

# Adding the input layer and the first hidden layer

# Adding the second hidden layer

# Adding the output layer


# Part 3 - Training the ANN
# Compiling the ANN

# Training the ANN on the Training set


# Part 4 - Making the predictions and evaluating the model
# Predicting the result of a single observation

# Predicting the Test set results

# Making the Confusion Matrix