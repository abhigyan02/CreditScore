import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

credit_data = pd.read_csv('credit_data.csv')

features = credit_data[['income', 'age', 'loan']]
target = credit_data.default

# machine learning handles arrays not data frames
X = np.array(features).reshape(-1, 3)
Y = np.array(target)

# 30% of data set is for testing and 70% of data set is for training
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(features_train, target_train)

predictions = model.fit.predict(features_test)

confusionMatrix = confusion_matrix(target_test, predictions, labels=model.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=model.classes_)
print("Confusion matrix: ")
display.plot()
plt.show()

print('Accuracy percentage', accuracy_score(target_test, predictions)*100, "%")


# using cross validation to predict a more general accuracy
prediction = cross_validate(model, X, Y, cv=5)
print("Cross Validation accuracy ", np.mean(prediction['test_score'])*100, "%")
