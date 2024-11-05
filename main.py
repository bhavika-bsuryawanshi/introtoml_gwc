# libraries and settings
import pandas as pd
import GWCutilities as util
pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)

print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# read the dataset
df = pd.read_csv("heartDisease_2020_sampling.csv")

# welcome message
print("welcome!\n\nwe will be performing data analysis on this 'indicators of heart disease' dataset.\n\na sample is shown below.")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

# print head of dataset
print(df.head())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
input("\npress enter to continue.\n")

# data cleaning

# "label encode" the dataset
df = util.labelEncoder(df, ["HeartDisease", "Smoking", "AlcoholDrinking", "Sex", "AgeCategory", "PhysicalActivity", "GenHealth"])

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\nbelow is a preview of the dataset after label encoding. \n")
print(df.head())
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
input("\npress enter to continue.\n")

# "one hot encode" the dataset
df = util.oneHotEncoder(df, ["Race"])

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\nbelow is a preview of the dataset after one hot encoding.\nthis will be the dataset used for data analysis. \n")

# print head of dataset
print(df.head())
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
input("\npress enter to continue.\n")

# create and train decision tree model
from sklearn.model_selection import train_test_split
X = df.drop("HeartDisease", axis = 1)
y = df["HeartDisease"]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(max_depth = 6, class_weight = "balanced")
clf = clf.fit(X_train, y_train)

# test the model and print accuracy score
test_predictions = clf.predict(X_test)
from sklearn.metrics import accuracy_score
test_acc = accuracy_score(y_test, test_predictions)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("\nthe accuracy with the testing data set of the decision tree is: " + str(test_acc) + ".")
print("\n")
      
# print confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, test_predictions, labels = [1,0])
print("the confusion matrix of the tree is:\n")
print(cm)

# train the model and print accuracy score
train_predictions = clf.predict(X_train)
from sklearn.metrics import accuracy_score
train_acc = accuracy_score(y_train, train_predictions)


print("\nthe accuracy with the training data set of the decision tree is: " + str(train_acc) + ".")

print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
input("\npress enter to continue.")
print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# application of decision trees and considerations
print("\ndecision trees can help narrow down a song (to sing, listen to, and/or play on an instrument) depending on various factors, such as the user's mood and free-time.\n\nhowever, when creating this decision tree, many factors must be overlooked to ensure the model performs fairly.\n\nthese include:\n- avoiding overfitting of the training dataset\n- ensuring that the features selected are important and not only specific to the training dataset\n- ensuring that the source of the training dataset is reliable, credible, and unbiased.\n")
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# prints a text version of the decision tree
input("\npress enter to continue.")
print("\nbelow is a text representation of how the decision tree makes choices:\n\n")
util.printTree(clf, X.columns)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
