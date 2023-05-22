import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

## Data load and split
data = pd.read_csv('cellular_churn_greece.csv')

X = data.drop(columns=['churned'])
print(X.head())
y = data['churned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

## Model training and scoring
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = clf.score(X_test, y_test)
precision, recall, fscore, support = precision_recall_fscore_support(y_test, y_pred, average='binary')

print(accuracy, precision, recall)

## Save the model and the data
filename = 'churn_model.pkl'
pickle.dump(clf, open(filename, 'wb'))

# Save X_test to a CSV file
X_test.to_csv('X_test.csv', index=False)

predictions = np.array(y_pred)

# Save predictions to a CSV file
np.savetxt('preds.csv', predictions, delimiter=',')