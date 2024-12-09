import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('regression/Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
print("Original X=\n", X, '\n')
print("Original y=\n", y, '\n')

y = y.reshape(len(y), 1) # we need to reshape the y array to a 2-D array, because the feature scaling class that we will use (StandardScaler) gets a 2-D array as an input
# 2 inputs of reshape function - > the first one: number of rows, the second: number of columns
print("2-D array y=\n", y, '\n')

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print('\nX Scaled =\n', X, '\n')
print('\ny Scaled =\n', y, '\n')

# Training the SVR model on the whole dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])).reshape(-1, 1)) # we should first scale the value (6.5), then the predict function will give us a scaled_y_pred (as y itself is scaled), then in order to have a y_pred in the original range, we should inverse the scaling transformation
print(y_pred)

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red') # we should give the original X, y, so we inverese the scaling transformation
plt.plot(sc_X.inverse_transform(X),
         sc_y.inverse_transform(regressor.predict(X).reshape(-1, 1)),
         color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color = 'red')
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid)).reshape(-1,1)), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()