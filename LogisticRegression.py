import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

credit_data = pd.read_csv('credit_data.csv')

features = credit_data[['income', 'age', 'loan']]
target = credit_data.default

# 30% of data set is for testing and 70% of data set is for training
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.3)

model = LogisticRegression()
model.fit = model.fit(features_train, target_train)

predictions = model.fit.predict(features_test)

print("Confusion matrix: ")
print( confusion_matrix(target_test, predictions))
print('Accuracy percentage', accuracy_score(target_test, predictions)*100, "%")
