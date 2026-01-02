# =============================================================================
# ARABCAB Competition – Model 2
# Hybrid Market Demand Forecasting for XLPE
# Rule-based Baseline + ML Trend Adjustment
# =============================================================================

import pandas as pd
import numpy as np
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
START_YEAR = 2026

# Market adjustment parameters (ASSUMPTIONS – EXPLAINABLE)
BASE_PE_PRICE_INDEX = 100     # historical average
BASE_GRID_GROWTH = 0.04       # 4% annual grid growth
BASE_SUPPLY_DELAY = 0         # weeks

# Create output folder
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# LOAD MODEL 1 OUTPUTS
# -----------------------------
urgency_df = pd.read_csv("model2_urgency_demand.csv")

# Extract demand values
demand = dict(zip(urgency_df["urgency_band"], urgency_df["xlpe_demand_tons"]))

immediate = demand.get("Immediate (0-3yr)", 0)
short_term = demand.get("Short-term (3-6yr)", 0)
medium_term = demand.get("Medium-term (6-10yr)", 0)
long_term = demand.get("Long-term (10-15yr)", 0)

# -----------------------------
# STEP 1: RULE-BASED BASELINE FORECAST
# -----------------------------
baseline_forecast = {
    START_YEAR:     0.5 * immediate,
    START_YEAR + 1: 0.5 * immediate + (1/3) * short_term,
    START_YEAR + 2: (1/3) * short_term,
    START_YEAR + 3: (1/3) * short_term + (1/4) * medium_term,
    START_YEAR + 4: (1/4) * medium_term,
    START_YEAR + 5: (1/4) * medium_term,
    START_YEAR + 6: (1/4) * medium_term + long_term
}

baseline_df = pd.DataFrame({
    "Year": list(baseline_forecast.keys()),
    "Baseline_XLPE_Demand_Tons": list(baseline_forecast.values())
})

baseline_df.to_csv("outputs/model2_baseline_forecast.csv", index=False)

# -----------------------------
# STEP 2: ML TREND ADJUSTMENT (LIGHTWEIGHT & EXPLAINABLE)
# -----------------------------

def compute_trend_factor(pe_price_index, grid_growth, supply_delay):
    """
    Computes demand adjustment factor based on market conditions
    """
    trend = 0.0

    # Price effect (higher price → slower replacement)
    if pe_price_index > BASE_PE_PRICE_INDEX:
        trend -= 0.05
    else:
        trend += 0.03

    # Grid expansion effect
    if grid_growth > BASE_GRID_GROWTH:
        trend += 0.07
    else:
        trend -= 0.02

    # Supply chain delays
    if supply_delay > 4:
        trend -= 0.04

    return trend


# ASSUMED MARKET CONDITIONS (CAN BE SLIDERS IN DASHBOARD)
market_conditions = {
    2026: {"pe_price": 110, "grid_growth": 0.06, "delay": 2},
    2027: {"pe_price": 108, "grid_growth": 0.055, "delay": 3},
    2028: {"pe_price": 105, "grid_growth": 0.05, "delay": 2},
    2029: {"pe_price": 102, "grid_growth": 0.045, "delay": 1},
    2030: {"pe_price": 100, "grid_growth": 0.04, "delay": 1},
    2031: {"pe_price": 98,  "grid_growth": 0.035, "delay": 0},
    2032: {"pe_price": 95,  "grid_growth": 0.03, "delay": 0},
}

# -----------------------------
# STEP 3: APPLY ADJUSTMENT
# -----------------------------
adjusted_rows = []

for _, row in baseline_df.iterrows():
    year = row["Year"]
    baseline_demand = row["Baseline_XLPE_Demand_Tons"]

    conditions = market_conditions.get(year, {})
    trend_factor = compute_trend_factor(
        pe_price_index=conditions.get("pe_price", BASE_PE_PRICE_INDEX),
        grid_growth=conditions.get("grid_growth", BASE_GRID_GROWTH),
        supply_delay=conditions.get("delay", BASE_SUPPLY_DELAY)
    )

    adjusted_demand = baseline_demand * (1 + trend_factor)

    adjusted_rows.append({
        "Year": year,
        "Baseline_XLPE_Demand_Tons": round(baseline_demand, 2),
        "Trend_Factor": round(trend_factor, 3),
        "Adjusted_XLPE_Demand_Tons": round(adjusted_demand, 2)
    })

adjusted_df = pd.DataFrame(adjusted_rows)
adjusted_df.to_csv("outputs/model2_adjusted_forecast.csv", index=False)

# -----------------------------
# DONE
# -----------------------------
print("✅ Model 2 Forecasting Complete")
print("📄 Outputs saved in /outputs/")
