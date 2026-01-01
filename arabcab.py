"""
================================================================================
MODEL 1: UTILITY CABLE HEALTH PREDICTION MODEL
================================================================================
ARABCAB Scientific Competition - AI-Based Demand Forecasting

PURPOSE:
    Predict cable health from utility inspection data using EXPLAINABLE models:
    1. Health Index (0-100) - Linear Regression
    2. Risk Level (Low/Medium/High/Critical) - Logistic Regression  
    3. Replacement Urgency (years) - Linear Regression

WHY LINEAR REGRESSION (Explainable AI)?
    - Coefficients show EXACTLY how each feature impacts predictions
    - Required for regulatory compliance in utilities
    - Judges can understand WHY the model makes decisions
    - Each coefficient = direct business insight
================================================================================
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_PATH = 'utility_inspection_data.csv'
MODEL_DIR = 'models'
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    'cable_age_years', 'cable_length_km', 'soil_corrosivity_index',
    'ambient_temp_C', 'humidity_percent', 'load_factor',
    'overload_events_last_year', 'fault_history_count', 'maintenance_score',
    'months_since_maintenance', 'partial_discharge_pC', 
    'insulation_resistance_MOhm', 'tan_delta',
]

CATEGORICAL_FEATURES = ['region', 'application', 'voltage_class']

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(filepath):
    print("\n" + "="*60)
    print("STEP 1: LOADING DATA")
    print("="*60)
    
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} records from {filepath}")
    
    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)
    
    # Get all feature columns
    feature_cols = NUMERIC_FEATURES.copy()
    for cat in CATEGORICAL_FEATURES:
        feature_cols.extend([c for c in df_encoded.columns if c.startswith(cat + '_')])
    
    print(f"✅ Features: {len(feature_cols)} (numeric + encoded categorical)")
    return df, df_encoded, feature_cols

# ============================================================================
# MODEL 1A: HEALTH INDEX (Linear Regression)
# ============================================================================
def train_health_model(df_encoded, feature_cols):
    print("\n" + "="*60)
    print("STEP 2: TRAINING HEALTH INDEX MODEL (Ridge Regression)")
    print("="*60)
    
    X = df_encoded[feature_cols]
    y = df_encoded['health_index']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    # Metrics
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'cv_r2': cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2').mean()
    }
    
    print(f"\n📊 HEALTH INDEX MODEL PERFORMANCE:")
    print(f"   R² Score:     {metrics['r2']:.4f}")
    print(f"   MAE:          {metrics['mae']:.2f} points")
    print(f"   RMSE:         {metrics['rmse']:.2f} points")
    print(f"   Cross-Val R²: {metrics['cv_r2']:.4f}")
    
    # Feature importance (coefficients)
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_,
        'abs_importance': np.abs(model.coef_)
    }).sort_values('abs_importance', ascending=False)
    
    print("\n📊 TOP 10 FEATURES (Explainable AI):")
    print("   (+ improves health, - degrades health)\n")
    for _, row in coef_df.head(10).iterrows():
        sign = "↑" if row['coefficient'] > 0 else "↓"
        print(f"   {sign} {row['feature']:<35} {row['coefficient']:>8.3f}")
    
    return model, scaler, metrics, coef_df

# ============================================================================
# MODEL 1B: RISK LEVEL (Logistic Regression)
# ============================================================================
def train_risk_model(df_encoded, feature_cols):
    print("\n" + "="*60)
    print("STEP 3: TRAINING RISK CLASSIFIER (Logistic Regression)")
    print("="*60)
    
    X = df_encoded[feature_cols]
    le = LabelEncoder()
    y = le.fit_transform(df_encoded['risk_level'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                         random_state=RANDOM_STATE, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 RISK CLASSIFIER ACCURACY: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    return model, scaler, le, accuracy

# ============================================================================
# MODEL 1C: REPLACEMENT URGENCY (Linear Regression)
# ============================================================================
def train_urgency_model(df_encoded, feature_cols):
    print("\n" + "="*60)
    print("STEP 4: TRAINING URGENCY MODEL (Ridge Regression)")
    print("="*60)
    
    X = df_encoded[feature_cols]
    y = df_encoded['replacement_urgency_years']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = Ridge(alpha=1.0, random_state=RANDOM_STATE)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    print(f"\n📊 URGENCY MODEL PERFORMANCE:")
    print(f"   R² Score: {metrics['r2']:.4f}")
    print(f"   MAE:      {metrics['mae']:.2f} years")
    print(f"   RMSE:     {metrics['rmse']:.2f} years")
    
    return model, scaler, metrics

# ============================================================================
# AGGREGATE DATA FOR MODEL 2 (Market Demand)
# ============================================================================
def aggregate_for_model2(df):
    print("\n" + "="*60)
    print("STEP 5: AGGREGATING DATA FOR MODEL 2 (Market Demand)")
    print("="*60)
    
    df['inspection_date'] = pd.to_datetime(df['inspection_date'])
    df['year'] = df['inspection_date'].dt.year
    df['quarter'] = df['inspection_date'].dt.quarter
    
    # Regional demand aggregation
    regional = df.groupby(['region', 'year', 'quarter']).agg({
        'xlpe_demand_tons': 'sum',
        'inspection_id': 'count',
        'health_index': 'mean',
        'replacement_urgency_years': 'mean'
    }).reset_index()
    regional.columns = ['region', 'year', 'quarter', 'total_xlpe_tons', 
                        'cable_count', 'avg_health', 'avg_urgency']
    
    # Urgency-based demand
    df['urgency_band'] = pd.cut(df['replacement_urgency_years'],
                                 bins=[0, 1, 3, 7, 15],
                                 labels=['Immediate', 'Short-term', 'Medium-term', 'Long-term'])
    
    urgency = df.groupby(['region', 'urgency_band']).agg({
        'xlpe_demand_tons': 'sum', 'inspection_id': 'count'
    }).reset_index()
    
    # Save for Model 2
    regional.to_csv('model2_regional_demand.csv', index=False)
    urgency.to_csv('model2_urgency_demand.csv', index=False)
    
    print("\n📊 REGIONAL XLPE DEMAND (Tons):")
    print(df.groupby('region')['xlpe_demand_tons'].agg(['sum', 'mean']).round(2).to_string())
    
    print("\n✅ Saved: model2_regional_demand.csv")
    print("✅ Saved: model2_urgency_demand.csv")
    
    return regional, urgency

# ============================================================================
# SAVE MODELS
# ============================================================================
def save_models(health_model, health_scaler, health_coef,
                risk_model, risk_scaler, risk_encoder,
                urgency_model, urgency_scaler, feature_cols):
    print("\n" + "="*60)
    print("STEP 6: SAVING MODELS")
    print("="*60)
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    
    # Health model
    joblib.dump(health_model, f'{MODEL_DIR}/health_regressor.pkl')
    joblib.dump(health_scaler, f'{MODEL_DIR}/health_scaler.pkl')
    
    # Risk model
    joblib.dump(risk_model, f'{MODEL_DIR}/risk_classifier.pkl')
    joblib.dump(risk_scaler, f'{MODEL_DIR}/risk_scaler.pkl')
    joblib.dump(risk_encoder, f'{MODEL_DIR}/risk_encoder.pkl')
    
    # Urgency model
    joblib.dump(urgency_model, f'{MODEL_DIR}/urgency_regressor.pkl')
    joblib.dump(urgency_scaler, f'{MODEL_DIR}/urgency_scaler.pkl')
    
    # Feature names and coefficients
    joblib.dump(feature_cols, f'{MODEL_DIR}/feature_names.pkl')
    health_coef.to_csv(f'{MODEL_DIR}/health_coefficients.csv', index=False)
    
    print("✅ All models saved to 'models/' directory")

# ============================================================================
# MAIN
# ============================================================================
def main():
    print("\n" + "="*60)
    print("  ARABCAB - MODEL 1: UTILITY HEALTH PREDICTION")
    print("  Explainable AI for Cable Condition Assessment")
    print("="*60)
    
    # Load data
    df, df_encoded, feature_cols = load_data(DATA_PATH)
    
    # Train models
    health_model, health_scaler, health_metrics, health_coef = train_health_model(df_encoded, feature_cols)
    risk_model, risk_scaler, risk_encoder, risk_acc = train_risk_model(df_encoded, feature_cols)
    urgency_model, urgency_scaler, urgency_metrics = train_urgency_model(df_encoded, feature_cols)
    
    # Aggregate for Model 2
    aggregate_for_model2(df)
    
    # Save models
    save_models(health_model, health_scaler, health_coef,
                risk_model, risk_scaler, risk_encoder,
                urgency_model, urgency_scaler, feature_cols)
    
    # Final summary
    print("\n" + "="*60)
    print("  MODEL 1 COMPLETE - SUMMARY")
    print("="*60)
    print(f"""
    ┌────────────────────────────────────────────────────────┐
    │  HEALTH INDEX MODEL (Ridge Regression)                 │
    │  • R²: {health_metrics['r2']:.4f}  • MAE: {health_metrics['mae']:.2f} pts              │
    ├────────────────────────────────────────────────────────┤
    │  RISK CLASSIFIER (Logistic Regression)                 │
    │  • Accuracy: {risk_acc:.2%}                               │
    ├────────────────────────────────────────────────────────┤
    │  URGENCY MODEL (Ridge Regression)                      │
    │  • R²: {urgency_metrics['r2']:.4f}  • MAE: {urgency_metrics['mae']:.2f} yrs              │
    └────────────────────────────────────────────────────────┘
    """)
    print("🎯 Model 1 outputs ready for Model 2 (Market Demand)")
    print("🎯 Dashboard can now use these models for predictions")
    
    return {'health': health_metrics, 'risk_acc': risk_acc, 'urgency': urgency_metrics}

if __name__ == "__main__":
    results = main()