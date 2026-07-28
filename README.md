# 🩺 AI Diabetes Risk Assessment & Machine Learning System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Logistic%20Regression-orange?style=for-the-badge&logo=scikit-learn)
![Frontend](https://img.shields.io/badge/Frontend-HTML5%20%2F%20CSS3%20%2F%20Vanilla%20JS-brightgreen?style=for-the-badge&logo=javascript)
![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)

An end-to-end Machine Learning solution for **Diabetes Risk Assessment and Clinical Evaluation** trained on the Pima Indians Diabetes Dataset (`diabetes.csv`). Features a hyperparameter-tuned `LogisticRegression` classifier, automated data noise removal, a comprehensive Jupyter Notebook with point-to-point theory, and an interactive dark-themed multi-page web application.

---

## ✨ Key Features

- **📊 Comprehensive Data Preprocessing & Noise Removal**:
  - Biologically impossible zero values in *Glucose*, *BloodPressure*, *SkinThickness*, *Insulin*, and *BMI* are filtered and imputed using group-wise medical mean imputation.
  - IQR (Interquartile Range) capping to eliminate extreme outlier noise without data loss.

- **🔬 Feature Engineering & Model Performance**:
  - Medical interaction terms (`Glucose_BMI`, `Age_Glucose`, `Insulin_Glucose`, log transforms).
  - **80:20 Stratified Train-Test Split** ($N_{train} = 614$, $N_{test} = 154$).
  - `GridSearchCV` hyperparameter optimization on `LogisticRegression`.
  - **87.01% Overall Test Accuracy** (ROC-AUC: **0.9159**).
  - **100% Accuracy Check** on random 10-sample verification test batch.

- **🌐 Multi-Page Glassmorphic Web Portal**:
  - **Step 1: Patient Personal Information** (Name, Age, Gender, Contact with `+91` prefix & 10-digit validation, Email, Primary Health Notes).
  - **Step 2: Clinical Parameters & Real-Time Risk Analysis** (Diagnostic inputs, probability gauge score, physiological risk breakdown).
  - **Step 3: Diabetic Care Plan & Precautions** (Triggered if diabetic risk detected — covers Immediate Clinical Actions, Low-GI Diet Plans, Aerobic Exercise Routines, Daily Glucose Tracking, and Hypoglycemia/Hyperglycemia Warning Signs).

- **📓 Point-to-Point Theory Jupyter Notebook**:
  - Complete `diabetes_analysis.ipynb` explaining EDA, noise filtering, model mathematics, cross-validation, and ROC-AUC curves in simple point-to-point theory.

---

## 🛠️ Technology Stack

- **Machine Learning & Core**: Python, Pandas, NumPy, Scikit-Learn, Joblib
- **Data Visualization**: Matplotlib, Seaborn
- **Backend Server**: Python HTTP REST Server
- **Frontend UI**: HTML5, Vanilla CSS3 (Glassmorphic Dark Theme), JavaScript (Canvas Animation Engine)

---

## 📂 Project Structure

```text
summer/
├── diabetes.csv             # Primary Dataset (Pima Indians Diabetes Dataset)
├── train_model.py           # ML Model Pipeline, Noise Filtration & Serialization Script
├── generate_notebook.py     # Programmatic Jupyter Notebook Generator
├── diabetes_analysis.ipynb  # Documented Source Code Notebook with Point-to-Point Theory
├── server.py                # Python HTTP Server & REST API Endpoint (/api/predict)
├── index.html               # Multi-Page Web Application Markup
├── style.css                # Dark Theme Glassmorphism Styling System
├── app.js                   # Canvas Animation Engine & Multi-Step Wizard Logic
├── model.pkl                # Trained Logistic Regression Weights
├── scaler.pkl               # StandardScaler Serialization File
├── features.json            # Feature Engineering Metadata & Means
├── start_app.bat            # One-Click Windows Launcher Batch File
├── run_project.bat          # Alternative Batch Launcher File
├── README.md                # Project Documentation
└── .gitignore               # Git Ignore Rules
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8 or higher installed on your system.
- Standard libraries: `pandas`, `numpy`, `scikit-learn`, `joblib`, `matplotlib`, `seaborn`.

To install dependencies:
```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn
```

### Running the Application

#### **Method 1: One-Click Launcher (Windows)**
Double-click **`start_app.bat`** (or **`run_project.bat`**) in the project folder.

#### **Method 2: Command Line**
1. Open your terminal in the project directory.
2. Execute:
   ```bash
   python server.py
   ```
3. Open your browser and navigate to:
   **`http://localhost:5000`**

---

## 📑 Machine Learning Results & Verification

| Evaluation Metric | Value |
| :--- | :--- |
| **Split Ratio** | 80:20 Stratified Split |
| **Overall Test Accuracy** | **87.01%** |
| **ROC-AUC Score** | **0.9159** |
| **Random 10-Sample Check** | **100.0% (10 / 10 Correct)** |

---

## 📜 License
This project is open source and available under the [MIT License](LICENSE).
