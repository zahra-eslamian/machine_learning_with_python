# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf

# Part 1 - Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('part8_deep_learning/artificial_neural_network/classification/Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values # eliminating the first 3 columns which don't help in model decision
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# feature scaling is absolutely necessary for deep learning, when creating an ANN
# So we do feature scaling for all the columns regradless of if some are already in the range of [0:1]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Building the ANN
# Initializing the ANN (as a sequesnce of layers, as opposed to a computational graph in which neurons are connected anyway and not in successive layers like Boltzmann Machines)
ann = tf.keras.models.Sequential()

# Adding the input layer and the first hidden layer
# The following line will create a fully connected layer (a shallow newtork and also automatically the input layer)
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
# 'units' is the number of neurons of the first shallow layer (not the nodes of the input layer which will be automatically picked based on our matrix of features X)
# 'activation': is the activation function we want to use & 'relu' is the code corressponding for Rectifier function

# Adding the second hidden layer
ann.add(tf.keras.layers.Dense(units=6, activation='relu')) # for each other hidden layers we want to add, we can consider any number of neurons. There is no any rule of thumb but it should be tunned as a hyper parameter

# Adding the output layer
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# In this dataset, we have one binary output so we need only one neuron for the output layer
# If we have a non-binary output like with 3 classes (A, B, C), we need 3 output neurons to onehot encode that dependent variable
# for the activation function, if we have non-binary classification, it should be 'softmax' and not sigmoid

# Part 3 - Training the ANN
# Compiling the ANN (with an optimizer to update the weights, a loss function and metrics like accuracy)
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# optimizer: 'adam' is the most common one which does Stochastic Gradient Descent
# loss: binary classification 'binary_crossentropy', non-binary classification 'categorical_crossentropy'
# metrics: we can add also other metrics and not only accuracy

# Training the ANN on the Training set
ann.fit(X_train, y_train, batch_size = 32, epochs = 100)
# batch_size: for batch learning (it's better not to pass each row one by one), the defualt is 32
# epochs: number of time the ANN gets trained on the whole training set

# Part 4 - Making the predictions and evaluating the model
# Predicting the result of a single observation
# Use our ANN model to predict if the customer with the following information will leave the bank: 
# Geography: France
# Credit Score: 600
# Gender: Male
# Age: 40 years old
# Tenure: 3 years
# Balance: \$ 60000
# Number of Products: 2
# Does this customer have a credit card? Yes
# Is this customer an Active Member: Yes
# Estimated Salary: \$ 50000
# So, should we say goodbye to that customer?

print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
# some points to keep in mind:
# 1. be careful that predict method needs a 2-D array as its argument
# 2. be careful for the dummy variables and put their corresponding encoded values
# 3. be careful to apply the same feature scaling that you did for the training set and the same median or standard deviation, so use 'transform' method instead of 'fit_transform' to avoid information leakage
# 4. the predict methon will return the probablity of the output, so we use the comparison and a threshold, here (0.5)

# Predicting the Test set results
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)