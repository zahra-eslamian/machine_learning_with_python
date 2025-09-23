import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset
dataset = pd.read_csv('Thyroid_Diff.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# X = dataset.drop('Recurred', axis=1)
# y = dataset['Recurred']
print("Original X: \n", X)
print("Original y: \n", y)


# Implement an instance of the ColumnTransformer class
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])], remainder='passthrough')

# Apply the fit_transform method on the instance of ColumnTransformer
X = ct.fit_transform(X)

# Convert the output into a NumPy array
X = np.array(X)

# Use LabelEncoder to encode binary categorical data
le = LabelEncoder()
y = le.fit_transform(y)

# Print the updated matrix of features and the dependent variable vector
print("Encoded matrix of features: \n", X)
print("Encoded dependent variable vector: \n", y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)

print('\nX_train\n', X_train)
print('\nX_test\n', X_test)
print('\ny_train\n', y_train)
print('\ny_test\n', y_test)

