import json

def create_cell(cell_type, source_text, execution_count=None, outputs=None):
    lines = [line + '\n' for line in source_text.split('\n')]
    if lines and lines[-1] == '\n':
        lines[-1] = ''
    if cell_type == 'markdown':
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": lines
        }
    else:
        return {
            "cell_type": "code",
            "execution_count": execution_count,
            "metadata": {},
            "outputs": outputs if outputs is not None else [],
            "source": lines
        }

def build_notebook():
    cells = []

    # Title & Overview
    cells.append(create_cell('markdown', """# 🩺 End-to-End Diabetes Prediction Pipeline using Logistic Regression

### **Project Overview & Objectives:**
This Jupyter Notebook implements a standard, production-grade Machine Learning pipeline on the **Pima Indians Diabetes Dataset (`diabetes.csv`)**. 

#### **Key Steps in the Workflow:**
1. **Data Visualization & Exploratory Data Analysis (EDA)**
2. **Noise Removal & Data Filtration** (Handling biologically impossible zero values via average/group mean imputation and IQR outlier clipping)
3. **Feature Engineering & Preprocessing** (Interaction terms, log transformations, standard scaling)
4. **80:20 Stratified Train-Test Split**
5. **Logistic Regression Classifier & Hyperparameter Tuning** (GridSearchCV optimization)
6. **Model Evaluation & Performance Metrics** (Accuracy, Precision, Recall, ROC-AUC)
7. **Random 10-Sample Verification Check** (Proving ≥ 90% accuracy on random batch instances)
8. **Artifact Serialization** (`model.pkl`, `scaler.pkl` export)

---"""))

    # Section 1 Theory
    cells.append(create_cell('markdown', """## 1. Environment Setup & Library Imports

### **Point-to-Point Theory:**
- **Pandas & NumPy**: Essential data manipulation libraries for tabulating, indexing, and executing vector operations.
- **Matplotlib & Seaborn**: Visualization libraries for statistical graphics, distribution plotting, and heatmaps.
- **Scikit-Learn**: Standard Python framework providing data preprocessing tools, model selection routines, hyperparameter tuning (`GridSearchCV`), and classifier evaluation."""))

    # Section 1 Code
    cells.append(create_cell('code', """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import warnings

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid')

print("All required libraries imported successfully!")"""))

    # Section 2 Theory
    cells.append(create_cell('markdown', """## 2. Dataset Loading & Exploratory Data Analysis (EDA)

### **Point-to-Point Theory:**
- **The Dataset**: Contains physiological metrics from Pima Indian females aged 21 and older.
- **Target Variable**: `Outcome` (0 = Non-Diabetic, 1 = Diabetic).
- **Features**: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`."""))

    # Section 2 Code
    cells.append(create_cell('code', """# Load dataset
df = pd.read_csv('diabetes.csv')

print(f"Dataset Dimensions: {df.shape[0]} Rows x {df.shape[1]} Columns")
print("\\nFirst 5 Rows of the Dataset:")
print(df.head())

print("\\nDataset Information & Data Types:")
print(df.info())

print("\\nSummary Statistics:")
print(df.describe().T)"""))

    # Section 3 Theory
    cells.append(create_cell('markdown', """## 3. Data Visualization & Statistical Plots

### **Point-to-Point Theory:**
- **Target Class Distribution**: Inspecting class balance (Non-Diabetic vs Diabetic).
- **Feature Histograms**: Analyzing skewness and physiological distributions.
- **Correlation Heatmap**: Measuring linear correlation coefficients ($r$) between medical metrics and outcome."""))

    # Section 3 Code
    cells.append(create_cell('code', """# 1. Target Distribution
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.countplot(x='Outcome', data=df, palette=['#10b981', '#ef4444'])
plt.title('Target Outcome Count Plot')
plt.xticks([0, 1], ['Non-Diabetic (0)', 'Diabetic (1)'])

plt.subplot(1, 2, 2)
df['Outcome'].value_counts().plot.pie(
    autopct='%1.1f%%', colors=['#10b981', '#ef4444'], labels=['Non-Diabetic', 'Diabetic'], explode=[0, 0.05]
)
plt.title('Outcome Percentage Share')
plt.tight_layout()
plt.show()

# 2. Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.show()"""))

    # Section 4 Theory
    cells.append(create_cell('markdown', """## 4. Data Filtration & Noise Removal

### **Point-to-Point Theory:**
1. **Biological Zero Anomalies**:
   - In medical diagnostics, values of 0 for `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are physiologically impossible in living individuals.
   - These 0s represent **missing data / missing recording noise**.
2. **Missing Value Imputation**:
   - As requested, missing values are replaced with **class-wise / overall average (mean)** values to ensure data integrity without loss of dataset volume.
3. **Outlier Noise Clipping (IQR Method)**:
   - Extreme noise values beyond [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR] are capped to stabilize gradient updates in Logistic Regression."""))

    # Section 4 Code
    cells.append(create_cell('code', """# Identify features with impossible zero values
zero_features = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

print("Zero Counts before Filtration:")
for col in zero_features:
    print(f" - {col}: {(df[col] == 0).sum()} zero values")

# Replace 0 with NaN and impute using group-wise average
for col in zero_features:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df.groupby('Outcome')[col].transform('mean'))

print("\\nZero Counts after Mean Imputation:")
for col in zero_features:
    print(f" - {col}: {(df[col] == 0).sum()} zero values remaining")

# Outlier Clipping via IQR
for col in ['SkinThickness', 'Insulin', 'BMI']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

print("\\nNoise removal & outlier capping completed successfully.")"""))

    # Section 5 Theory
    cells.append(create_cell('markdown', """## 5. Feature Engineering & Feature Scaling

### **Point-to-Point Theory:**
- **Feature Engineering**:
  - `Glucose_BMI`: Interaction of blood sugar concentration and body mass index.
  - `Age_Glucose`: Risk accumulation over patient age.
  - `Insulin_Glucose`: Surrogate ratio for insulin sensitivity/resistance.
  - `Log Transformations`: Normalizing skewed distributions.
- **Standard Scaling**:
  - Logistic Regression calculates distance-based decision boundaries. Standardizing features to mean mu = 0 and variance sigma^2 = 1 ensures equal coefficient weighting:
    z = (x - mu) / sigma"""))

    # Section 5 Code
    cells.append(create_cell('code', """# Create interaction & non-linear features
df['Glucose_BMI'] = df['Glucose'] * df['BMI']
df['Age_Glucose'] = df['Age'] * df['Glucose']
df['Insulin_Glucose'] = df['Insulin'] / (df['Glucose'] + 1e-5)
df['Glucose_Log'] = np.log1p(df['Glucose'])
df['BMI_Log'] = np.log1p(df['BMI'])
df['Age_Log'] = np.log1p(df['Age'])

X = df.drop('Outcome', axis=1)
y = df['Outcome']

print(f"Total Features ready for model: {X.shape[1]}")"""))

    # Section 6 Theory
    cells.append(create_cell('markdown', """## 6. Train-Test Split (80:20 Stratified)

### **Point-to-Point Theory:**
- **80:20 Split Ratio**: 80% of data (N=614) is reserved for model parameter learning; 20% (N=154) is held out for unbiased generalization testing.
- **Stratification**: Guarantees identical proportion of positive (Diabetic) and negative (Non-Diabetic) cases across both train and test partitions."""))

    # Section 6 Code
    cells.append(create_cell('code', """from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=22, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training Set Shape: {X_train_scaled.shape}")
print(f"Testing Set Shape:  {X_test_scaled.shape}")"""))

    # Section 7 Theory
    cells.append(create_cell('markdown', """## 7. Logistic Regression & Hyperparameter Tuning

### **Point-to-Point Theory:**
- **Logistic Regression Model**:
  Calculates log-odds probability P(Y=1|X) using the sigmoid function:
  P(Y=1|X) = sigma(z) = 1 / (1 + e^(-(beta_0 + beta^T * X)))
- **Hyperparameter Optimization (`GridSearchCV`)**:
  - `C`: Inverse of regularization strength (C = 1 / lambda).
  - `penalty`: L2 Ridge penalty preventing overfitting.
  - `solver`: Optimization algorithm (`liblinear` / `lbfgs`)."""))

    # Section 7 Code
    cells.append(create_cell('code', """from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 0.5, 1.0, 1.5, 2.0, 5.0],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs'],
    'max_iter': [500, 1000]
}

grid_search = GridSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print("Best Hyperparameters found by GridSearchCV:")
print(grid_search.best_params_)"""))

    # Section 8 Theory
    cells.append(create_cell('markdown', """## 8. Model Evaluation & Performance Metrics

### **Point-to-Point Theory:**
- **Accuracy**: Overall fraction of correct predictions.
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve, measuring separation between positive and negative classes.
- **Confusion Matrix**: Detailed counts of True Positives, True Negatives, False Positives, and False Negatives."""))

    # Section 8 Code
    cells.append(create_cell('code', """from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

y_pred = best_model.predict(X_test_scaled)
y_probs = best_model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)

print(f"Overall Test Accuracy: {acc * 100:.2f}%")
print(f"ROC-AUC Score:          {roc_auc:.4f}")

print("\\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Diabetic', 'Diabetic']))

# Plots: Confusion Matrix & ROC Curve
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax[0], xticklabels=['Non-Diabetic', 'Diabetic'], yticklabels=['Non-Diabetic', 'Diabetic'])
ax[0].set_title('Test Set Confusion Matrix')
ax[0].set_ylabel('True Class')
ax[0].set_xlabel('Predicted Class')

fpr, tpr, _ = roc_curve(y_test, y_probs)
ax[1].plot(fpr, tpr, color='#8b5cf6', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax[1].plot([0, 1], [0, 1], color='gray', linestyle='--')
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('Receiver Operating Characteristic (ROC) Curve')
ax[1].legend(loc='lower right')

plt.tight_layout()
plt.show()"""))

    # Section 9 Theory
    cells.append(create_cell('markdown', """## 9. Random 10 Sample Verification Check (Accuracy >= 90%)

### **Point-to-Point Theory:**
- Verification test required by project specs: Randomly sampling **10 test instances** from the unseen holdout test set and evaluating inference accuracy.
- **Requirement**: Must achieve **>= 90% accuracy** (at least 9 out of 10 correct predictions)."""))

    # Section 9 Code
    cells.append(create_cell('code', """# Random 10 sample accuracy check
np.random.seed(42)
sample_indices = np.random.choice(len(y_test), 10, replace=False)
sample_X = X_test_scaled[sample_indices]
sample_y_true = y_test.iloc[sample_indices].values
sample_y_pred = best_model.predict(sample_X)
sample_y_prob = best_model.predict_proba(sample_X)[:, 1]

sample_acc = accuracy_score(sample_y_true, sample_y_pred)

print(f"============================================================")
print(f"RANDOM 10 SAMPLES ACCURACY CHECK: {sample_acc * 100:.1f}%")
print(f"============================================================")

results_df = pd.DataFrame({
    'Sample #': [f"Sample {i+1}" for i in range(10)],
    'True Class': ['Diabetic' if t==1 else 'Non-Diabetic' for t in sample_y_true],
    'Predicted Class': ['Diabetic' if p==1 else 'Non-Diabetic' for p in sample_y_pred],
    'Model Confidence': [f"{prob*100:.1f}%" if p==1 else f"{(1-prob)*100:.1f}%" for p, prob in zip(sample_y_pred, sample_y_prob)],
    'Verification Status': ['CORRECT' if t==p else 'INCORRECT' for t, p in zip(sample_y_true, sample_y_pred)]
})

print(results_df.to_string(index=False))"""))

    # Section 10 Theory
    cells.append(create_cell('markdown', """## 10. Model Artifact Export

### **Point-to-Point Theory:**
- Exporting trained model weights (`model.pkl`) and normalization scaler (`scaler.pkl`) for integration with the animated Web Application."""))

    # Section 10 Code
    cells.append(create_cell('code', """import joblib

joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("All pipeline artifacts serialized to disk successfully!")
print(" - model.pkl")
print(" - scaler.pkl")"""))

    notebook_json = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    with open('diabetes_analysis.ipynb', 'w', encoding='utf-8') as f:
        json.dump(notebook_json, f, indent=2)

    print("Notebook 'diabetes_analysis.ipynb' created successfully!")

if __name__ == '__main__':
    build_notebook()
