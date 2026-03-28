import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.explainability import generate_mix_explanation

app = Flask(__name__)

# Load trained XGBoost model
model = joblib.load("models/ConcreteAI_XGBoost_Best.joblib")

# Feature order must exactly match train.py
FEATURE_COLUMNS = [
    "Cement_kg_m3",
    "Fly_Ash_kg_m3",
    "GGBS_kg_m3",
    "metakolin_kg_m3",
    "Water_kg_m3",
    "Sand_kg_m3",
    "AGE",
    "admixture",
    "Coarse aggregate",
    "SCMContent",
    "Binder",
    "WBRatio",
    "AggregateToBinder",
    "AdmixtureToBinder",
]

# Cost factors
COSTS = np.array([
    6.0,    # Cement
    2.0,    # Fly Ash
    3.6,    # GGBS
    8.0,    # Metakaolin
    0.0,    # Water
    0.9,    # Sand
    45.0,   # Admixture
    1.05    # Coarse Aggregate
])

# CO2 factors
CO2_FACTORS = np.array([
    1.008,   # Cement
    0.026,   # Fly Ash
    0.064,   # GGBS
    0.45,    # Metakaolin
    0.0003,  # Water
    0.006,   # Sand
    0.72,    # Admixture
    0.014    # Coarse Aggregate
])

def build_feature_map(cement, flyash, ggbs, metakaolin, water, admixture, coarse_agg, sand, age):
    scm_content = flyash + ggbs + metakaolin
    binder = cement + scm_content
    total_aggregate = sand + coarse_agg

    binder_safe = binder if binder != 0 else 1e-8

    wb_ratio = water / binder_safe
    aggregate_to_binder = total_aggregate / binder_safe
    admixture_to_binder = admixture / binder_safe



    return {
        "Cement_kg_m3": cement,
        "Fly_Ash_kg_m3": flyash,
        "GGBS_kg_m3": ggbs,
        "metakolin_kg_m3": metakaolin,
        "Water_kg_m3": water,
        "Sand_kg_m3": sand,
        "AGE": age,
        "admixture": admixture,
        "Coarse aggregate": coarse_agg,
        "SCMContent": scm_content,
        "Binder": binder,
        "WBRatio": wb_ratio,
        "AggregateToBinder": aggregate_to_binder,
        "AdmixtureToBinder": admixture_to_binder,
        "TotalAggregate": total_aggregate
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        cement = float(data["cement"])
        flyash = float(data["flyash"])
        ggbs = float(data["ggbs"])
        metakaolin = float(data["metakaolin"])
        water = float(data["water"])
        admixture = float(data["admixture"])
        coarse_agg = float(data["coarse_agg"])
        sand = float(data["fine_agg"])
        age = float(data["days"])

        feature_map = build_feature_map(
            cement=cement,
            flyash=flyash,
            ggbs=ggbs,
            metakaolin=metakaolin,
            water=water,
            admixture=admixture,
            coarse_agg=coarse_agg,
            sand=sand,
            age=age
        )

        feature_df = pd.DataFrame([feature_map])[FEATURE_COLUMNS]

        strength = float(model.predict(feature_df)[0])

        quantities = np.array([
            cement,
            flyash,
            ggbs,
            metakaolin,
            water,
            sand,
            admixture,
            coarse_agg
        ])

        cost = float(np.sum(quantities * COSTS))
        co2 = float(np.sum(quantities * CO2_FACTORS))

        explanation_text = generate_mix_explanation(feature_map)

        return jsonify({
            "strength": round(strength, 2),
            "cost": round(cost, 2),
            "co2": round(co2, 2),
            "explanation_text": explanation_text
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == "__main__":
    app.run(debug=True)