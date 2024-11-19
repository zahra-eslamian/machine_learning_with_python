# Data Preprocessing Tools

# Importing the libraries
import numpy as np  # to working with arrays
import matplotlib.pyplot as plt # plot charts and graphs
import pandas as pd # import the dataset and create the metrics and features

# Importing the dataset
# iloc = locate indexes
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values # the features (independant variables) matrix
y = dataset.iloc[:, -1].values  # the dependant variable vector
print('X =\n', X, '\n')
print('y =\n', y)

# Taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #np here means numpy/ nan:missing data, and we choose avg (or mean) as the strategy
imputer.fit(X[:, 1:3])  
#fit method from the class 'SimpleImputer':
# connects the imputer to the features matrix and it receives as an argument all the columns of X with numerical values
X[:, 1:3] = imputer.transform(X[:, 1:3]) #transform method fills the missing data with the mean (the strategy we specified) of the values of the related column
print('\nMissing data addressed\nX:\n', X)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#arguments of ColumnTransformer: transformers:which transformer we want to use and on which columns
# remainder:what we want to do with the other columns, here 'passthrough' means do nothing, we have other options like 'drop'
X = np.array(ct.fit_transform(X)) # we need to convert the output to a numpy array, because to train the model, we will use a train function called 'fit' which expects the features matrix as a numpy array
print('\nCategorical features encoded\nX:\n',X)

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print('\nCategorical dependant variable encoded\ny:\n',y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
#we will have 4 sets because we need a pair of matrix of features and dependant variable for the Training set and another pair for the Test set
# the parameter 'random_state' here is just for teaching purposes to gain exactly the same training and test sets (us and the instructor)
print('\nX_train\n', X_train)
print('\nX_test\n', X_test)
print('\ny_train\n', y_train)
print('\ny_test\n', y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])     #we need to apply the same scaling of the trainin set, so we will not calculate new mean and std dev
print('\nX_train_scaled\n', X_train)
print('\nX_test_scaled\n', X_test)





########### SOME HINTS ################
# Missing data in the DataFrame can be identified using pandas methods like isnull and sum.
# Consider using df.isnull().sum() to get the number of missing values in each column.

