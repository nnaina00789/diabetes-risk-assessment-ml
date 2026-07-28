import http.server
import socketserver
import json
import os
import joblib
import numpy as np
import urllib.parse
import webbrowser
import subprocess
import threading
import time
import sys
import csv
import io

PORT = 5000

# Load trained ML model, scaler, and metadata
MODEL_PATH = 'model.pkl'
SCALER_PATH = 'scaler.pkl'
METADATA_PATH = 'features.json'

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Loaded model.pkl and scaler.pkl successfully!")
else:
    model = None
    scaler = None
    print("WARNING: Model artifacts not found. Run train_model.py first.")

if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
else:
    metadata = {}

def compute_feature_vector(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, means):
    # Biological zero check & mean fallback
    if glucose <= 0: glucose = means['Glucose']
    if bp <= 0: bp = means['BloodPressure']
    if skin <= 0: skin = means['SkinThickness']
    if insulin <= 0: insulin = means['Insulin']
    if bmi <= 0: bmi = means['BMI']

    glucose_bmi = glucose * bmi
    age_glucose = age * glucose
    insulin_glucose = insulin / (glucose + 1e-5)
    glucose_log = np.log1p(glucose)
    bmi_log = np.log1p(bmi)
    age_log = np.log1p(age)

    return np.array([[
        pregnancies, glucose, bp, skin, insulin, bmi, dpf, age,
        glucose_bmi, age_glucose, insulin_glucose, glucose_log, bmi_log, age_log
    ]])

class MLHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_POST(self):
        # 1. Single Patient Prediction Endpoint (/api/predict)
        if self.path == '/api/predict':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                data = json.loads(post_data.decode('utf-8'))
                
                pregnancies = float(data.get('pregnancies', 1))
                glucose = float(data.get('glucose', 120))
                bp = float(data.get('bloodPressure', 72))
                skin = float(data.get('skinThickness', 29))
                insulin = float(data.get('insulin', 155))
                bmi = float(data.get('bmi', 32.5))
                dpf = float(data.get('dpf', 0.47))
                age = float(data.get('age', 33))

                means = metadata.get('means', {
                    'Glucose': 121.69, 'BloodPressure': 72.41,
                    'SkinThickness': 29.15, 'Insulin': 155.55, 'BMI': 32.46
                })

                raw_vec = compute_feature_vector(pregnancies, glucose, bp, skin, insulin, bmi, dpf, age, means)
                scaled_features = scaler.transform(raw_vec)

                prediction = int(model.predict(scaled_features)[0])
                probabilities = model.predict_proba(scaled_features)[0]
                diabetic_prob = float(probabilities[1])
                non_diabetic_prob = float(probabilities[0])

                if diabetic_prob < 0.35:
                    risk_tier = "Low Risk"
                    risk_color = "#10b981"
                elif diabetic_prob < 0.65:
                    risk_tier = "Moderate Risk"
                    risk_color = "#f59e0b"
                else:
                    risk_tier = "High Risk"
                    risk_color = "#ef4444"

                # Feature contributions calculation (Feature Explainability)
                all_feats = metadata.get('all_features', [])
                coefs = metadata.get('coefficients', {})
                contributions = []

                if coefs and len(all_feats) == scaled_features.shape[1]:
                    # Calculate raw feature risk impact
                    raw_impacts = {
                        'Glucose': abs(scaled_features[0][1] * coefs.get('Glucose', 0) + scaled_features[0][8] * coefs.get('Glucose_BMI', 0) + scaled_features[0][11] * coefs.get('Glucose_Log', 0)),
                        'BMI': abs(scaled_features[0][5] * coefs.get('BMI', 0) + scaled_features[0][8] * coefs.get('Glucose_BMI', 0) + scaled_features[0][12] * coefs.get('BMI_Log', 0)),
                        'Age': abs(scaled_features[0][7] * coefs.get('Age', 0) + scaled_features[0][9] * coefs.get('Age_Glucose', 0) + scaled_features[0][13] * coefs.get('Age_Log', 0)),
                        'Insulin': abs(scaled_features[0][4] * coefs.get('Insulin', 0) + scaled_features[0][10] * coefs.get('Insulin_Glucose', 0)),
                        'Pedigree Score': abs(scaled_features[0][6] * coefs.get('DiabetesPedigreeFunction', 0)),
                        'Blood Pressure': abs(scaled_features[0][2] * coefs.get('BloodPressure', 0)),
                        'Skin Thickness': abs(scaled_features[0][3] * coefs.get('SkinThickness', 0)),
                        'Pregnancies': abs(scaled_features[0][0] * coefs.get('Pregnancies', 0))
                    }

                    total_impact = sum(raw_impacts.values()) + 1e-5
                    for feat_name, impact_val in sorted(raw_impacts.items(), key=lambda x: x[1], reverse=True):
                        pct = round((impact_val / total_impact) * 100, 1)
                        contributions.append({"feature": feat_name, "percentage": pct})

                # Risk factors breakdown
                risk_factors = []
                if glucose >= 140:
                    risk_factors.append(f"Elevated Fasting Glucose ({glucose:.0f} mg/dL) - Exceeds normal range (70-99 mg/dL)")
                if bmi >= 30:
                    risk_factors.append(f"High BMI ({bmi:.1f} kg/m²) - Indicates Obesity classification (≥30 kg/m²)")
                if age >= 45:
                    risk_factors.append(f"Age ({age:.0f} yrs) - Increased age-related risk factor")
                if dpf >= 0.8:
                    risk_factors.append(f"High Genetic Pedigree Score ({dpf:.2f}) - Strong family history indicator")
                if bp >= 90:
                    risk_factors.append(f"Elevated Blood Pressure ({bp:.0f} mmHg) - Stage 1 Hypertension risk")

                response = {
                    "success": True,
                    "prediction": prediction,
                    "label": "Diabetic Risk Detected" if prediction == 1 else "Non-Diabetic Profile",
                    "diabetic_probability": round(diabetic_prob * 100, 1),
                    "non_diabetic_probability": round(non_diabetic_prob * 100, 1),
                    "risk_tier": risk_tier,
                    "risk_color": risk_color,
                    "feature_contributions": contributions,
                    "risk_factors": risk_factors if risk_factors else ["All primary physiological markers are within expected normal ranges."],
                    "clinical_notes": [
                        "Fasting blood glucose should be re-tested after 8 hours of fasting.",
                        "Regular physical exercise (150 mins/week) significantly lowers diabetes risk.",
                        "Maintain balanced dietary carbohydrate intake and consult a physician."
                    ]
                }

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))

            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        # 2. Bulk CSV Prediction Endpoint (/api/predict_csv)
        elif self.path == '/api/predict_csv':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            
            try:
                csv_text = post_data.decode('utf-8')
                reader = csv.DictReader(io.StringIO(csv_text))
                
                means = metadata.get('means', {
                    'Glucose': 121.69, 'BloodPressure': 72.41,
                    'SkinThickness': 29.15, 'Insulin': 155.55, 'BMI': 32.46
                })

                output_rows = []
                for i, row in enumerate(reader):
                    pid = row.get('Patient_ID', f"PAT-{1000 + i}")
                    preg = float(row.get('Pregnancies', 1))
                    gluc = float(row.get('Glucose', 120))
                    bp = float(row.get('BloodPressure', 72))
                    skin = float(row.get('SkinThickness', 29))
                    ins = float(row.get('Insulin', 155))
                    bmi = float(row.get('BMI', 32.5))
                    dpf = float(row.get('DiabetesPedigreeFunction', row.get('DPF', 0.47)))
                    age = float(row.get('Age', 33))

                    raw_vec = compute_feature_vector(preg, gluc, bp, skin, ins, bmi, dpf, age, means)
                    scaled = scaler.transform(raw_vec)

                    pred = int(model.predict(scaled)[0])
                    prob = float(model.predict_proba(scaled)[0][1]) * 100

                    tier = "Low Risk" if prob < 35 else ("Moderate Risk" if prob < 65 else "High Risk")
                    label = "Diabetic" if pred == 1 else "Non-Diabetic"

                    output_rows.append({
                        "Patient_ID": pid,
                        "Glucose": gluc,
                        "BMI": bmi,
                        "Age": age,
                        "Prediction_Label": label,
                        "Risk_Probability_%": round(prob, 1),
                        "Risk_Tier": tier
                    })

                # Write result CSV
                output_buffer = io.StringIO()
                writer = csv.DictWriter(output_buffer, fieldnames=["Patient_ID", "Glucose", "BMI", "Age", "Prediction_Label", "Risk_Probability_%", "Risk_Tier"])
                writer.writeheader()
                writer.writerows(output_rows)

                csv_output = output_buffer.getvalue()

                self.send_response(200)
                self.send_header('Content-Type', 'text/csv')
                self.send_header('Content-Disposition', 'attachment; filename=diabetes_predictions_result.csv')
                self.end_headers()
                self.wfile.write(csv_output.encode('utf-8'))

            except Exception as e:
                self.send_response(400)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

        else:
            super().do_GET()

class ReusableTCPServer(socketserver.TCPServer):
    allow_reuse_address = True

def open_default_browser(url):
    time.sleep(0.8)
    try:
        if hasattr(os, 'startfile'):
            os.startfile(url)
        else:
            webbrowser.open(url)
    except Exception:
        webbrowser.open(url)

def run_server():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    handler = MLHandler
    server_url = f"http://localhost:{PORT}"

    threading.Thread(target=open_default_browser, args=(server_url,), daemon=True).start()

    try:
        with ReusableTCPServer(("", PORT), handler) as httpd:
            print("=========================================================================")
            print(f"   DIABETES ASSESSMENT SERVER IS LIVE AT: {server_url}")
            print("=========================================================================")
            print("Opening default web browser...")
            print("Press Ctrl+C to stop the server.\n")
            httpd.serve_forever()
    except OSError as e:
        if e.winerror == 10048:
            print(f"\n[INFO] Server is ALREADY active on: {server_url}")
            print("[INFO] Opening browser now...\n")
            open_default_browser(server_url)
        else:
            raise e

if __name__ == '__main__':
    run_server()
