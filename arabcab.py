import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import joblib  # <--- NEW: For saving the model
import os

# 1. LOAD DATA 
df = pd.read_csv('utility_cable_data.csv') 

# 2. FEATURE ENGINEERING
X = df[['cable_age', 'electricity_usage', 'pe_price_lag_30', 'lead_time_days']]
y = df['xlpe_demand_tons']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. TRAIN THE AI
model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/cable_xgboost_model.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print("Model saved to models/cable_xgboost_model.pkl")