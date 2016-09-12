#
# In this and the following exercises, you'll be adding train test splits to the data
# to see how it changes the performance of each classifier
#
# The code provided will load the Titanic dataset like you did in project 0, then train
# a decision tree (the method you used in your project) and a Bayesian classifier (as
# discussed in the introduction videos). You don't need to worry about how these work for
# now. 
#
# What you do need to do is import a train/test split, train the classifiers on the
# training data, and store the resulting accuracy scores in the dictionary provided.

import numpy as np
import pandas as pd

# Load the dataset
X = pd.read_csv('titanic_data.csv')
T=pd.read_csv('titanic_data.csv')
# Limit to numeric data
X = X._get_numeric_data()
# Separate the labels
y = X['Survived']
# Remove labels from the inputs, and age due to missing data
del X['Age'], X['Survived']

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import cross_validation
# TODO: split the data into training and testing sets,
# using the standard settings for train_test_split.
# Then, train and test the classifiers with your newly split data instead of X and y.

def predictions_3(data):
    """ Model with multiple features. Makes a prediction with an accuracy of at least 80%. """
    
    predictions = []
    for index,row in data.iterrows():
        if(row['Sex']=='female'):
            if((row['Age']>=40 and row['Age']<=50) and row['Pclass']==3):
                predictions.append(0);
            else:
                predictions.append(1);
        
        
        elif(row['Sex']=='male' and row['Age']<10):
            predictions.append(1);
        else:
            predictions.append(0);
    
    # Return our predictions
    return pd.Series(predictions)
# Make the predictions
y = predictions_3(T)

# The decision tree classifier
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=0.4, random_state=0)
clf1 = DecisionTreeClassifier()
clf1.fit(X_train,y_train)
print "Decision Tree has accuracy: ",accuracy_score(y_test,clf1.predict(X_test))
dtc=accuracy_score(clf1.predict(X),y)
print clf1.predict(X)
# The naive Bayes classifier
'''

clf2 = GaussianNB()
clf2.fit(X_train,y_train)
print "GaussianNB has accuracy: ",accuracy_score(y_test,clf2.predict(X_test))
nb=accuracy_score(clf2.predict(X),y)

answer = { 
 "Naive Bayes Score": nb, 
 "Decision Tree Score": dtc
}'''