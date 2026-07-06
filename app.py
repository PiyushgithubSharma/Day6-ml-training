from pathlib import Path

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model_path = Path(__file__).resolve().parent / "models" / "linear_regression_model.joblib"
model = joblib.load(model_path)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    area = float(request.form["area"])
    bedrooms = int(request.form["bedrooms"])
    bathrooms = int(request.form["bathrooms"])
    stories = int(request.form["stories"])
    parking = int(request.form["parking"])

    mainroad = request.form["mainroad"]
    guestroom = request.form["guestroom"]
    basement = request.form["basement"]
    hotwaterheating = request.form["hotwaterheating"]
    airconditioning = request.form["airconditioning"]

    prefarea = request.form["prefarea"]
    furnishingstatus = request.form["furnishingstatus_furnished"]

    binary = {
        "yes": 1,
        "no": 0
    }

    furnishing_status_mapping = {
        "furnished": [1, 0, 0],
        "semi-furnished": [0, 1, 0],
        "unfurnished": [0, 0, 1],
    }

    furnishing_status_features = furnishing_status_mapping[furnishingstatus]

    feature = [
        area,
        bedrooms,
        bathrooms,
        stories,
        binary[mainroad],
        binary[guestroom],
        binary[basement],
        binary[hotwaterheating],
        binary[airconditioning],
        parking,
        binary[prefarea],
        furnishing_status_features[0],
        furnishing_status_features[1],
        furnishing_status_features[2],
    ]

  
    prediction = model.predict([feature])[0]

    return render_template(
        "index.html",
        prediction=f"Estimated Price : ₹ {prediction:,.0f}"
    )


if __name__ == "__main__":
    app.run(debug=True)