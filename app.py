# import libraries
import numpy as np
from flask import Flask, request, render_template
import pickle
import os
import pandas as pd
from collections import OrderedDict

# app name
app = Flask(__name__)

# load the saved model
def load_model():
    return pickle.load(open('models\\loan_xgc_model.pkl','rb'))

# home page
@app.route('/')
def home():
    return render_template('index.html')

# predict the results and return it to frontend

def create_example(values):
    dict = OrderedDict()
    property_dict = OrderedDict()
    cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area']

    for col, value in zip(cols, values):
            dict[col] = float(value)

    df = pd.DataFrame(dict, index=[0])

    return df


@app.route('/predict', methods=['POST'])
def predict():

    labels = ['Approved', 'Rejected']

    features = [x for x in request.form.values()]

    example = create_example(features)

    model = load_model()

    prediction = model.predict(example)

    result = labels[prediction[0]]

    return render_template('index.html', output='Your loan is {}'.format(result))


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
