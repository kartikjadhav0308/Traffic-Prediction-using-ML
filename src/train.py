"""
Model Training & Evaluation Script
Trains Random Forest and XGBoost Regressors on traffic.csv dataset.
"""

import os
import json
import time
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

try:
    from preprocess import load_and_preprocess_dataset, FEATURE_COLUMNS
except ImportError:
    from src.preprocess import load_and_preprocess_dataset, FEATURE_COLUMNS


def train_models(data_path: str = 'traffic.csv', models_dir: str = 'models'):
    """
    Trains Random Forest and XGBoost models, evaluates their performance,
    and serializes the models and evaluation metrics to disk.
    """
    os.makedirs(models_dir, exist_ok=True)

    print(f"[*] Loading dataset: '{data_path}'...")
    X, y = load_and_preprocess_dataset(data_path)
    print(f"[*] Total Records : {len(X):,}")
    print(f"[*] Features ({len(FEATURE_COLUMNS)}) : {FEATURE_COLUMNS}")

    # Train / Test split (80% training, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"[*] Training Samples: {len(X_train):,} | Testing Samples: {len(X_test):,}")

    # ---------------- 1. Random Forest Regressor ----------------
    print("\n" + "=" * 50)
    print("Training Random Forest Regressor (100 trees)...")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_time = time.time() - t0

    rf_pred = rf.predict(X_test)
    rf_mae = float(mean_absolute_error(y_test, rf_pred))
    rf_rmse = float(np.sqrt(mean_squared_error(y_test, rf_pred)))
    rf_r2 = float(r2_score(y_test, rf_pred))

    rf_path = os.path.join(models_dir, 'random_forest_model.joblib')
    joblib.dump(rf, rf_path)
    print(f"[+] Random Forest trained in {rf_time:.2f}s and saved to {rf_path}")
    print(f"    - R2 Score : {rf_r2:.4f} ({rf_r2 * 100:.2f}%)")
    print(f"    - MAE      : {rf_mae:.2f} vehicles")
    print(f"    - RMSE     : {rf_rmse:.2f} vehicles")

    # ---------------- 2. XGBoost Regressor ----------------
    print("\n" + "=" * 50)
    print("Training XGBoost Regressor (150 estimators)...")
    t0 = time.time()
    xgb = XGBRegressor(
        n_estimators=150,
        max_depth=7,
        learning_rate=0.08,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_time = time.time() - t0

    xgb_pred = xgb.predict(X_test)
    xgb_mae = float(mean_absolute_error(y_test, xgb_pred))
    xgb_rmse = float(np.sqrt(mean_squared_error(y_test, xgb_pred)))
    xgb_r2 = float(r2_score(y_test, xgb_pred))

    xgb_path = os.path.join(models_dir, 'xgboost_model.joblib')
    joblib.dump(xgb, xgb_path)
    print(f"[+] XGBoost trained in {xgb_time:.2f}s and saved to {xgb_path}")
    print(f"    - R2 Score : {xgb_r2:.4f} ({xgb_r2 * 100:.2f}%)")
    print(f"    - MAE      : {xgb_mae:.2f} vehicles")
    print(f"    - RMSE     : {xgb_rmse:.2f} vehicles")

    # Feature Importance Calculations
    rf_feat_imp = {f: round(float(imp), 4) for f, imp in zip(FEATURE_COLUMNS, rf.feature_importances_)}
    xgb_feat_imp = {f: round(float(imp), 4) for f, imp in zip(FEATURE_COLUMNS, xgb.feature_importances_)}

    metrics = {
        'dataset_source': 'Kaggle Traffic Prediction Dataset (traffic.csv)',
        'total_samples': len(X),
        'features': FEATURE_COLUMNS,
        'random_forest': {
            'r2': round(rf_r2, 4),
            'mae': round(rf_mae, 2),
            'rmse': round(rf_rmse, 2),
            'train_time_sec': round(rf_time, 2),
            'feature_importance': dict(sorted(rf_feat_imp.items(), key=lambda x: x[1], reverse=True))
        },
        'xgboost': {
            'r2': round(xgb_r2, 4),
            'mae': round(xgb_mae, 2),
            'rmse': round(xgb_rmse, 2),
            'train_time_sec': round(xgb_time, 2),
            'feature_importance': dict(sorted(xgb_feat_imp.items(), key=lambda x: x[1], reverse=True))
        }
    }

    metrics_path = os.path.join(models_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[+] Metrics saved to: '{metrics_path}'")
    print("=" * 50)


if __name__ == '__main__':
    train_models()
