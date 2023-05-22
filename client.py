import requests
import pandas as pd
import numpy as np

X_test = pd.read_csv('X_test.csv')
loaded_predictions = np.loadtxt('preds.csv', delimiter=',')

url = 'http://127.0.0.1:5000/predict_churn'

# Define the observations
observations = list()
samples = np.random.choice(range(1, X_test.shape[0]-1), size=5, replace=False)

for i in samples:
    observation = X_test.iloc[i]
    response = requests.get(url, params=observation.to_dict())
    prediction = response.text
    test = (str(prediction) == str(int(loaded_predictions[i])))
    print(f"prediction match expected value? {test}")