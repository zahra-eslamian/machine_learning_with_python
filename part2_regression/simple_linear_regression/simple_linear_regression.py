# Importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('regression/simple_linear_regression/Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
#instead of the above line, we can also write as follows:
# plt.plot(X_test, y_pred, color = 'blue')
#but as the equation (formula) that we use for regerssion is the same, there would be no difference
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Making a single prediction (for example the salary of an employee with 12 years of experience)
print(regressor.predict([[12]]))
#Notice that the value of the feature (12 years) was input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting 12 into a double pair of square brackets makes the input exactly a 2D array

#Get the final values of the coefficients b0 and b1 for the regression equation y = b0 + b1 x 
print('Slope Coefficient (b1): ', regressor.coef_)
print('y-intercept (b0): ', regressor.intercept_)

#in R there is the method summary() which we can call on our regressor to retrieve some statistical analysis regarding the accuracy of the model and how much the independent variable affects the dependent variable...in python there seems not to be such thing in scikit-learn for regression, but you can use the package statsmodels
