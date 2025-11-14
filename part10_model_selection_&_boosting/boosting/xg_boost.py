# Run the following command in the terminal to install the apyori package: pip install xgboost

# Importing the libraries
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('part10_model_selection_&_boosting/boosting/Breast_Cancer.csv')
X = dataset.iloc[:, :-1] # keep as DataFrame
y = dataset.iloc[:, -1] # keep as Series
# .values converts the column into a NumPy array. So, we 
# Then later, when training, XGBoost will automatically convert to NumPy.
# XGBoost wants the class labels to be 0 and 1, but in the Wisconsin Breast Cancer dataset the labels are 2 (benign) and 4 (malignant).
y = y.map({2: 0, 4: 1})

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Training XGBoost on the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))