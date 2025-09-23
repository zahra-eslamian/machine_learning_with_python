# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Data Preprocessing
dataset = pd.read_csv('part5_association_rule_learning/Market_Basket_Optimisation.csv', header = None) # without setting this 'header' argument to None, pandas will think that the first row is the name of the columns, but in this .csv file, the first row is actually the first transaction.
transactions = []
for i in range(0, 7501):
    # apyori does not accept the pandas dataframe, but we need to change it to a list
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)]) 

# Training the Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
# the value for min_support/min_confidence/min_liftin depends on the business, the rule of thumb for min_lift is suggested to be at least 3
# in this case, we set min/max_length to 2, because we want the rule of buy one, get one free

# Visualising the results
# Displaying the first results coming directly from the output of the apriori function
results = list(rules)
print('\n', results)

# Putting the results well organised into a Pandas DataFrame
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))

resultsinDataFrame = pd.DataFrame(
            inspect(results),
            columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift']
        )

# Displaying the results non sorted
print('\n', resultsinDataFrame)

# Displaying the results sorted by descending lifts
print('\n', resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))


