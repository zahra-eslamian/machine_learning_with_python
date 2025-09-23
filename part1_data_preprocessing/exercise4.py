# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the Iris dataset
df = pd.read_csv('iris.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Separate features and target
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
target = ['target']

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print('\nX_train\n', X_train)
print('\nX_test\n', X_test)
print('\ny_train\n', y_train)
print('\ny_test\n', y_test)

# Apply feature scaling on the training and test sets
sc = StandardScaler()
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])

# Print the scaled training and test sets
print(X_train)
print(X_test)


####### Instructor Solution
# Load the Iris dataset using pd.read_csv
iris_df = pd.read_csv('iris.csv')

# Separate features and target
X = iris_df.drop('target', axis=1)  # Assuming 'target' is the column name for the target variable
y = iris_df['target']

# Code from geeksforgeeks
# x = df.drop('target',axis=1) 
# y = df[['target']]

# Split the dataset into an 80-20 training-test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply StandardScaler to scale the feature variables
# The StandardScaler is applied to standardize the features to have a mean=0 and variance=1. The scaler is fitted on the training set and then used to transform both the training and test sets.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Print scaled training and test sets
print("Scaled Training Set:")
print(X_train)
print("\nScaled Test Set:")
print(X_test)