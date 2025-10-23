# ***** Importing the libraries *****
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# ***** Importing the dataset *****
dataset = pd.read_csv('part7_natural_language_processing/Restaurant_Reviews.tsv', delimiter = '\t',
                      quoting = 3) # as we have quoting text (like: "yum yum sauce"), we set quoting parameter here to 3, which means "no quoting", to perevent processing errors


# ***** Cleaning the texts *****
# Regular expression operations library of Python, it allows us to deal with special characters
import re

# The Natural Language Toolkit (NLTK) is a Python package for natural language processing. We need to install it first: pip install nltk
import nltk
# download all words which does not help us to decide whether a review is positive or negative, like articles, pronouns, ...
nltk.download('stopwords') 
# import the stopwords
from nltk.corpus import stopwords

# keep only the root of words to optimize and minimize the dimenation of our sparse matrix (the number of columns). Ex) loved, loves, Love -> Love
from nltk.stem.porter import PorterStemmer

# define a list to contain all the cleaned reviews
corpus = []

for i in range(0, 1000):
    # replace all the punctuations (anything which is not a letter) with space
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])

    # convert all the uppercase letters to lowercase
    review = review.lower()

    # split each review to words
    review = review.split() 

    # apply stemming to each word of a review, if that word is not among English stopwords
    ps = PorterStemmer()
    # the word 'not' stranegly is considered a stopword in nltk library, but definitely 'not' is an important word in deciding whether a review is positive of negative, so we should remove it from the list of stopwords
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]

    # join back again all the words of a review, but also adding an spsce between them
    review = ' '.join(review)

    # fill our corpus list with cleaned reviews
    corpus.append(review)

# print(corpus)


# ***** Creating the Bag of Words model *****
# tokenization: to create the sparse matrix containing all the reviews in different rows, and all the words of all the reviews in different collumns where the cell will get a '1' if the review contains that word, or a '0' if not
from sklearn.feature_extraction.text import CountVectorizer
# for creating an instance of CountVectorizer class, we should specify a parameter which is acutally the maximum number of words/columns, but we can postpone it till after we create our matrix of features (X)
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray() # the fit method will take all the words for our corpus, and the transform method will transform the words to columns...and also we use the toarray function to convert the X array into a 2-D array as algorithms like Naive Bayes accepts 2-D arrays as their input
y = dataset['Liked']

print(len(X[0])) # output: 1566, now we can set that "max_features" for instantiating an obj of         
                 # CountVectorizer class like to 1500 most frequent words


# ***** Splitting the dataset into the Training set and Test set *****
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
print("\nX Train: ", X_train)
print("\ny Train: ", y_train)
print("\nX Test: ", X_test)
print("\ny Test: ", y_test)


# ***** Training the Naive Bayes model on the Training set *****
# we chose Naive Bayes as it usually works well with NLP problems
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# ***** Predicting the Test set results *****
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.to_numpy().reshape(len(y_test),1)),1)) # to display next to each other the vector of predictions and the vector of real results
# equivalent to the above line:
# print(np.concatenate(
#     (y_pred.reshape(-1,1), y_test.values.reshape(-1,1)),
#     axis=1
# ))
# another equivalent
# print(np.column_stack((y_pred, y_test.to_numpy())))


# ***** Calculating the Confusion Matrix and Accuracy *****
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))


# ***** Predicting if a single review is positive or negative *****
# Use our model to predict if the review "I love this restaurant so much" is positive or negative.
# Solution: We just repeat the same text preprocessing process we did before, but this time with a single review.
new_review = 'I love this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower()
new_review = new_review.split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)