import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
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

## Evaluate Training
'''confusion matrix visualisation to make it easier to view for myself and understand since the print of numbers was messing me up'''
classes = [0,1,2]
cm = confusion_matrix(y_test, y_pred, labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap=plt.cm.Blues)
plt.title("confusion matrix", fontsize=15, pad=20)
plt.xlabel("prediction", fontsize=11)
plt.ylabel("actual", fontsize=11)
plt.gca().xaxis.set_label_position('top')
plt.gca().xaxis.tick_top()
plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()


''' Accuracy measuring and classification notes'''
# accuracy = accuracy_score(y_test, y_pred)
# classification_rprt = classification_report(y_test, y_pred)
# print(f"\nAccuracy: {accuracy:.2f}")
# print("\nClassification Report:\n", classification_rprt)