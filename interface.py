import pandas as pd
import numpy as np
import pickle
from flask import Flask
from flask import request

app = Flask(__name__)


def read_model(filename, X_test_file, predictions_file):
    model = pickle.load(open(filename, 'rb'))
    X_test = pd.read_csv(X_test_file)
    y_pred = np.loadtxt(predictions_file, delimiter=',')

    # Perform predictions on X_test using the loaded model
    predictions = model.predict(X_test)

    # Compare the predictions with the loaded predictions
    predictions_match = np.array_equal(predictions, y_pred)
    print(f'Match between the model and the dataset? {predictions_match}')
    return model


@app.route('/predict_churn', methods=['GET'])
def predict_churn():
    data = request.args
    row_data = pd.DataFrame(data, index=[0])
    return str(model.predict(row_data)[0])


if __name__ == '__main__':
    model = read_model('churn_model.pkl', 'X_test.csv', 'preds.csv')
    app.run(debug=True)