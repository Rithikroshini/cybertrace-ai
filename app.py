import os
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ---------------------------------------------------------
# 1. ATMS DATASET (20 Real Chennai Regions - Illustrative Coordinates)
# ---------------------------------------------------------
ATMS = [
    {"id": "ATM_01", "name": "SBI ATM - Central Station", "lat": 13.0827, "lng": 80.2707, "type": "Metro Transit Hub"},
    {"id": "ATM_02", "name": "HDFC ATM - T. Nagar Commercial", "lat": 13.0418, "lng": 80.2341, "type": "Commercial Market"},
    {"id": "ATM_03", "name": "ICICI ATM - Velachery Main Road", "lat": 12.9759, "lng": 80.2212, "type": "Suburban Tech Belt"},
    {"id": "ATM_04", "name": "Axis Bank - Anna Nagar Blue Star", "lat": 13.0850, "lng": 80.2101, "type": "Residential Highway"},
    {"id": "ATM_05", "name": "Canara Bank - Guindy Industrial", "lat": 13.0067, "lng": 80.2020, "type": "Industrial Zone"},
    {"id": "ATM_06", "name": "Indian Bank - Tambaram Junction", "lat": 12.9249, "lng": 80.1000, "type": "Transit Junction"},
    {"id": "ATM_07", "name": "Kotak Bank - Adyar Signal", "lat": 13.0012, "lng": 80.2565, "type": "Urban Intersection"},
    {"id": "ATM_08", "name": "PNB ATM - Koyambedu Bus Terminus", "lat": 13.0694, "lng": 80.1948, "type": "Metro Transit Hub"},
    {"id": "ATM_09", "name": "BOB ATM - OMR Perungudi Tech Park", "lat": 12.9654, "lng": 80.2480, "type": "Suburban Tech Belt"},
    {"id": "ATM_10", "name": "Union Bank - Porur Junction", "lat": 13.0382, "lng": 80.1565, "type": "Transit Junction"},
    {"id": "ATM_11", "name": "SBI ATM - Chennai Airport Terminal", "lat": 12.9815, "lng": 80.1636, "type": "Metro Transit Hub"},
    {"id": "ATM_12", "name": "Axis Bank - Nungambakkam High Rd", "lat": 13.0624, "lng": 80.2429, "type": "Commercial Market"},
    {"id": "ATM_13", "name": "HDFC ATM - Chromepet GST Road", "lat": 12.9516, "lng": 80.1409, "type": "Residential Highway"},
    {"id": "ATM_14", "name": "ICICI ATM - Sholinganallur Signal", "lat": 12.9010, "lng": 80.2279, "type": "Suburban Tech Belt"},
    {"id": "ATM_15", "name": "Indian Overseas Bank - Mylapore Tank", "lat": 13.0332, "lng": 80.2697, "type": "Commercial Market"},
    {"id": "ATM_16", "name": "Canara Bank - Egmore Station", "lat": 13.0783, "lng": 80.2611, "type": "Metro Transit Hub"},
    {"id": "ATM_17", "name": "SBI ATM - Vadapalani Bus Depot", "lat": 13.0500, "lng": 80.2121, "type": "Transit Junction"},
    {"id": "ATM_18", "name": "Bank of India - Saidapet West", "lat": 13.0213, "lng": 80.2231, "type": "Urban Intersection"},
    {"id": "ATM_19", "name": "HDFC ATM - Ambattur OT Junction", "lat": 13.1142, "lng": 80.1548, "type": "Industrial Zone"},
    {"id": "ATM_20", "name": "ICICI ATM - Royapettah Clock Tower", "lat": 13.0543, "lng": 80.2642, "type": "Commercial Market"}
]

SCAM_MAP = {"phishing": 0, "investment": 1, "parttime": 2, "vishing": 3}
TYPE_BOOST = {
    "Metro Transit Hub": 0.35, "Commercial Market": 0.25, "Suburban Tech Belt": 0.10,
    "Residential Highway": 0.20, "Industrial Zone": 0.05, "Transit Junction": 0.30, 
    "Urban Intersection": 0.15
}
NIGHT_TYPES = ("Metro Transit Hub", "Transit Junction", "Residential Highway")

# ---------------------------------------------------------
# 2. FEATURE-BASED SYNTHETIC TARGET SELECTION & TRAINING
# ---------------------------------------------------------
def pick_atm(vlat, vlng, hour, amount):
    scores = []
    for a in ATMS:
        # Distance in km approx
        d = np.hypot((a["lat"] - vlat) * 111, (a["lng"] - vlng) * 111 * np.cos(np.radians(vlat)))
        s = np.exp(-d / 5.0) + TYPE_BOOST[a["type"]]
        
        # Night-time transit preference
        if (hour >= 22 or hour < 5) and a["type"] in NIGHT_TYPES:
            s += 0.25
            
        # Large cash out prefers secluded/commercial hubs
        if amount > 100000 and a["type"] in ("Commercial Market", "Industrial Zone"):
            s += 0.15
            
        scores.append(s + np.random.uniform(0, 0.12))
        
    return ATMS[int(np.argmax(scores))]["id"]

# Generate dataset
np.random.seed(42)
num_samples = 1000

v_lats = np.random.uniform(12.90, 13.12, num_samples)
v_lngs = np.random.uniform(80.10, 80.28, num_samples)
amounts = np.random.choice([15000, 35000, 75000, 150000, 300000], num_samples)
scam_codes = np.random.choice([0, 1, 2, 3], num_samples)
hours = np.random.randint(0, 24, num_samples)

targets = [pick_atm(lat, lng, hr, amt) for lat, lng, hr, amt in zip(v_lats, v_lngs, hours, amounts)]

df = pd.DataFrame({
    'victim_lat': v_lats,
    'victim_lng': v_lngs,
    'amount': amounts,
    'scam_type_code': scam_codes,
    'incident_hour': hours,
    'target_atm': targets
})

X = df[['victim_lat', 'victim_lng', 'amount', 'scam_type_code', 'incident_hour']]
y = df['target_atm']

Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
ml_model = RandomForestClassifier(n_estimators=100, random_state=42)
ml_model.fit(Xtr, ytr)

accuracy = ml_model.score(Xte, yte)
print(f"✅ Random Forest Trained! Holdout Evaluation Accuracy (Synthetic): {accuracy * 100:.2f}%")

# ---------------------------------------------------------
# 3. ROUTES & API ENDPOINTS
# ---------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        v_lat = float(data.get('victim_lat', 13.0500))
        v_lng = float(data.get('victim_lng', 80.2400))
        amount = float(data.get('amount', 75000))
        scam_type = data.get('scam_type', 'phishing')
        incident_hour = int(data.get('incident_hour', 14))

        scam_code = SCAM_MAP.get(scam_type, 0)
        input_features = [[v_lat, v_lng, amount, scam_code, incident_hour]]

        probabilities = ml_model.predict_proba(input_features)[0]
        classes = list(ml_model.classes_)

        predictions = []
        for atm in ATMS:
            if atm["id"] in classes:
                idx = classes.index(atm["id"])
                risk_score = round(probabilities[idx] * 100, 2)
            else:
                risk_score = 0.0

            predictions.append({
                "id": atm["id"],
                "name": atm["name"],
                "lat": atm["lat"],
                "lng": atm["lng"],
                "type": atm["type"],
                "risk_score": risk_score
            })

        predictions = sorted(predictions, key=lambda x: x['risk_score'], reverse=True)

        # Feature Importance Explanation for Judges
        importances = ml_model.feature_importances_
        feature_names = ['Spatial Coordinates', 'Coordinates (Lng)', 'Amount Size', 'Scam Type', 'Time of Incident']
        top_factor_idx = int(np.argmax(importances))
        top_factor = feature_names[top_factor_idx]

        return jsonify({
            "status": "success",
            "predictions": predictions,
            "top_factor": top_factor,
            "model_accuracy": round(accuracy * 100, 1)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=False, port=5000)