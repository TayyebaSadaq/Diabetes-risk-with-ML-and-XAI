from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
# import models from train-models.py

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
accuracy = accuracy_score(y_test, y_pred)
classification_rprt = classification_report(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_rprt)