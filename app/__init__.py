import os
from flask import Flask, make_response, jsonify, request
from config import configs, Config
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)
app.config.from_object(configs['production'])



model = pickle.load(open('model.pkl', 'rb'))

@app.route("/api/weather", methods=['POST'])
def predict_weather():
    # get the data from the post request
    data = request.get_json(force=True)

    #make predictions using model loaded from disk as per the data
    prediction =  model.predict([[data]])

    # Take the first value of prediction
    output = prediction[0]

    return jsonify(output)