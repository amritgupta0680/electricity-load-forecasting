from flask import Flask, render_template, request
import joblib
import numpy as np
import datetime

app = Flask(__name__)

# ===============================
# LOAD MODELS
# ===============================
household_model = joblib.load(
    r"C:\Users\rakhi\Desktop\energy_forecasting_flask\models\household_hourly_BEST.joblib"
)

city_model = joblib.load(
    r"C:\Users\rakhi\Desktop\energy_forecasting_flask\models\city_hourly_TOTAL.joblib"
)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    level = "household"

    if request.method == "POST":
        level = request.form["level"]

        # ===============================
        # USER INPUTS (RAW kWh)
        # ===============================
        lag_1_raw = float(request.form["lag_1"])
        lag_24_raw = float(request.form["lag_24"])
        rolling_24_raw = float(request.form["rolling_24"])

        # ===============================
        # LOG TRANSFORM INPUTS
        # (MATCH TRAINING EXACTLY)
        # ===============================
        lag_1 = np.log1p(lag_1_raw)
        lag_24 = np.log1p(lag_24_raw)
        lag_168 = lag_24

        rolling_24_mean = np.log1p(rolling_24_raw)
        rolling_168_mean = rolling_24_mean

        # ===============================
        # TIME FEATURES
        # ===============================
        now = datetime.datetime.now()

        hour = now.hour
        day = now.day
        month = now.month
        day_of_week = now.weekday()
        is_weekend = 1 if day_of_week >= 5 else 0

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        dow_sin = np.sin(2 * np.pi * day_of_week / 7)
        dow_cos = np.cos(2 * np.pi * day_of_week / 7)

        # ===============================
        # FINAL FEATURE VECTOR
        # (EXACT ORDER FROM TRAINING)
        # ===============================
        X = np.array([[
            hour,
            day,
            month,
            day_of_week,
            is_weekend,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            lag_1,
            lag_24,
            lag_168,
            rolling_24_mean,
            rolling_168_mean
        ]])

        # ===============================
        # PREDICT (LOG SPACE)
        # ===============================
        if level == "household":
            log_pred = household_model.predict(X)[0]
        else:
            log_pred = city_model.predict(X)[0]

        # ===============================
        # INVERSE LOG TRANSFORM
        # ===============================
        prediction = np.expm1(log_pred)

    return render_template(
        "index.html",
        prediction=prediction,
        level=level
    )

if __name__ == "__main__":
    app.run(debug=True)
