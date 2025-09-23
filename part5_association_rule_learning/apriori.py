# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('part5_association_rule_learning/Market_Basket_Optimisation.csv')
X = dataset.iloc[:, [3, 4]].values