import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

## dataset import
data = pd.read_csv("data/diabetes.csv")

## seperate features and target variable
X = data.drop(columns=['Diabetes_012']) # features
y = data['Diabetes_012'] # target

## split data into training and testing sets for RF model
""" 
    80 % of data being used to train model (X_train, y_train)
    20% being used for testing (test_size)
    random_state=42 means we get the same split each time
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## building random forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)

## evaluate accuracy
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix: \n")
print(conf_matrix)
accuracy = accuracy_score(y_test, y_pred)
classification_rprt = classification_report(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rprt)