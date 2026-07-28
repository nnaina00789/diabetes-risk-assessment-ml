import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def train_and_save_model():
    print("=" * 60)
    print("1. LOADING DATASET")
    print("=" * 60)
    df = pd.read_csv('diabetes.csv')
    print(f"Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")

    print("\n" + "=" * 60)
    print("2. NOISE REMOVAL & DATA FILTRATION (MISSING VALUE IMPUTATION)")
    print("=" * 60)
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    for col in zero_cols:
        zero_count = (df[col] == 0).sum()
        print(f"Feature '{col}': {zero_count} invalid zero values found.")
        df[col] = df[col].replace(0, np.nan)
        col_mean = df[col].mean()
        df[col] = df[col].fillna(df.groupby('Outcome')[col].transform('mean'))
        print(f"  -> Imputed NaN values in '{col}' using group average (overall mean: {col_mean:.2f})")

    # Outlier capping using IQR
    for col in ['SkinThickness', 'Insulin', 'BMI']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    print("  -> Noise removal & outlier clipping completed successfully.")

    print("\n" + "=" * 60)
    print("3. FEATURE ENGINEERING")
    print("=" * 60)
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Age_Glucose'] = df['Age'] * df['Glucose']
    df['Insulin_Glucose'] = df['Insulin'] / (df['Glucose'] + 1e-5)
    df['Glucose_Log'] = np.log1p(df['Glucose'])
    df['BMI_Log'] = np.log1p(df['BMI'])
    df['Age_Log'] = np.log1p(df['Age'])
    print(f"Engineered features added. Total features now: {df.shape[1] - 1}")

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    print("\n" + "=" * 60)
    print("4. TRAIN-TEST SPLIT (80:20 STRATIFIED)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=22, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set:  {X_test.shape[0]} samples (20%)")

    print("\n" + "=" * 60)
    print("5. FEATURE SCALING")
    print("=" * 60)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Feature scaling applied using StandardScaler.")

    print("\n" + "=" * 60)
    print("6. LOGISTIC REGRESSION HYPERPARAMETER TUNING")
    print("=" * 60)
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
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    print("\n" + "=" * 60)
    print("7. MODEL EVALUATION ON TEST SET")
    print("=" * 60)
    y_pred = best_model.predict(X_test_scaled)
    y_probs = best_model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    print(f"Overall Test Accuracy: {acc * 100:.2f}%")
    print(f"ROC-AUC Score:          {roc_auc:.4f}")

    print("\n" + "=" * 60)
    print("8. ACCURACY CHECK ON RANDOM 10 SAMPLE BATCH")
    print("=" * 60)
    np.random.seed(42)
    sample_indices = np.random.choice(len(y_test), 10, replace=False)
    sample_X = X_test_scaled[sample_indices]
    sample_y_true = y_test.iloc[sample_indices].values
    sample_y_pred = best_model.predict(sample_X)
    sample_acc = accuracy_score(sample_y_true, sample_y_pred)
    print(f"Random 10 Samples Accuracy: {sample_acc * 100:.1f}%")

    print("\n" + "=" * 60)
    print("9. SAVING MODEL, SCALER, AND COEFFICIENTS")
    print("=" * 60)
    joblib.dump(best_model, 'model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    feature_names = list(X.columns)
    coefficients = best_model.coef_[0].tolist()
    intercept = float(best_model.intercept_[0])

    metadata = {
        'raw_features': list(X.columns[:8]),
        'all_features': feature_names,
        'coefficients': dict(zip(feature_names, coefficients)),
        'intercept': intercept,
        'accuracy': float(acc),
        'roc_auc': float(roc_auc),
        'random_10_accuracy': float(sample_acc),
        'means': {col: float(df[col].mean()) for col in zero_cols}
    }
    with open('features.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("Model ('model.pkl'), Scaler ('scaler.pkl'), and Metadata ('features.json') exported successfully!")

if __name__ == '__main__':
    train_and_save_model()
