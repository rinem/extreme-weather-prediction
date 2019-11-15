import json
import pandas as pd
import time
import numpy as np
import itertools
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix
import scipy 
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import scale
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from flask import Flask,render_template,url_for,request

app = Flask(__name__)


def preprocess(state, year):
    data = pd.read_csv('Data/'+state+'.csv')
    temp = data[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
       'AUG', 'SEP', 'OCT', 'NOV', 'DEC']].loc[data['YEAR'] == year]

    data_year = np.asarray(temp[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
        'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])

    X_year = None; y_year = None
    for i in range(data_year.shape[1]-3):
        if X_year is None:
            X_year = data_year[:, i:i+3]
            y_year = data_year[:, i+3]
        else:
            X_year = np.concatenate((X_year, data_year[:, i:i+3]), axis=0)
            y_year = np.concatenate((y_year, data_year[:, i+3]), axis=0)
    return (X_year, y_year)

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        year = int(data['Year'])
        fname = data['State']
        p_in = open("trained/"+fname+".pkl","rb")
        model = pickle.load(p_in)
        x = pd.read_csv("test/"+fname+"_input.csv")
        a, b, c = float(x[x['YEAR'] == year]['Mar-May']), float(x[x['YEAR'] == year]['avgjune']),float(x[x['YEAR'] == year]['sub'])
        l = [[a, b, c]]
        my_prediction = model.predict(l)
        my_prediction = my_prediction[0]
        result = {}
    if my_prediction == '0':
        result['data'] = "Very Low Rainfall"
    elif my_prediction == '-1':
        result['data'] = "Light to Medium Rainfall"
    elif my_prediction == '1':
        result['data'] = "Heavy Rainfall"
    print(result)
    return result

@app.route('/rainfall', methods=['POST'])
def rainfall():
    if request.method == 'POST':
        data = request.get_json()
        year = int(data['Year'])
        fname = data['State']
        x_test, y_test = preprocess(fname, year)
        p_in = open("graphs/"+fname+".pkl","rb")
        model = pickle.load(p_in)
        y_pred = model.predict(x_test)
        result = {}
        result['actual'] = list(y_test)
        result['prediction'] = list(y_pred)
        print(result)
        return result


if __name__ == '__main__':
	app.run(debug=True)

