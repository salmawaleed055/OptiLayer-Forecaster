"""
================================================================================
ARABCAB Competition - Data Preprocessing
================================================================================
Converts the real 15-KV XLPE Cable data into the format needed for Model 1.
Adds derived features for demand forecasting.
================================================================================
"""

import pandas as pd
import numpy as np

def load_and_preprocess_real_data(filepath='15-KV XLPE Cable.xlsx'):
    """
    Load the real utility cable inspection data and preprocess it.
    
    Input columns:
        - ID: Cable identifier
        - Age: Cable age in years
        - Partial Discharge: PD measurement (0-1 normalized)
        - Visual Condition: Good/Medium/Poor
        - Neutral Corrosion: Corrosion index (0-1)
        - Loading: Load value
        - Health Index: 1-5 scale (1=worst, 5=best)
    
    Output: Preprocessed DataFrame ready for Model 1
    """
    
    print("=" * 60)
    print("ARABCAB - Loading Real Cable Data")
    print("=" * 60)
    
    # Load Excel file
    df = pd.read_excel(filepath)
    
    # Clean column names (remove any whitespace)
    df.columns = df.columns.str.strip()
    
    print(f"✅ Loaded {len(df)} cable records from {filepath}")
    print(f"✅ Columns: {df.columns.tolist()}")
    
    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    
    # 1. Convert Visual Condition to numeric (for ML)
    condition_map = {'Good': 1, 'Medium': 2, 'Poor': 3}
    df['visual_condition_score'] = df['Visual Condition'].map(condition_map)
    
    # 2. Create Risk Level based on Health Index (1-5 scale)
    def get_risk_level(health):
        if health >= 5:
            return 'Low'
        elif health >= 4:
            return 'Medium'
        elif health >= 3:
            return 'High'
        else:  # 1 or 2
            return 'Critical'
    
    df['risk_level'] = df['Health Index'].apply(get_risk_level)
    
    # 3. Calculate Replacement Urgency (years)
    # Lower health = more urgent replacement
    # Scale: Health 1 → 1 year, Health 5 → 15 years
    df['replacement_urgency_years'] = (df['Health Index'] / 5) * 15
    df['replacement_urgency_years'] = df['replacement_urgency_years'].clip(0.5, 15).round(1)
    
    # 4. Estimate XLPE Demand (tons)
    # Based on: cables needing replacement sooner = higher immediate demand
    # Assume average cable length ~2km for 15kV distribution cables
    cable_length_km = 2.0
    voltage_multiplier = 1.5  # MV cable (15kV is medium voltage)
    
    urgency_factor = (15 - df['replacement_urgency_years']) / 15  # 0 to 1
    df['xlpe_demand_tons'] = (
        cable_length_km * voltage_multiplier * 0.5 * (1 + urgency_factor)
    ).round(2)
    
    # 5. Normalize Health Index to 0-100 scale for consistency
    df['health_index_100'] = (df['Health Index'] / 5 * 100).round(1)
    
    print(f"\n📊 Data Summary:")
    print(f"   Age range: {df['Age'].min()} - {df['Age'].max()} years")
    print(f"   Health Index distribution:")
    print(df['Health Index'].value_counts().sort_index().to_string())
    print(f"\n   Risk Level distribution:")
    print(df['risk_level'].value_counts().to_string())
    
    return df

def create_model2_inputs(df):
    """
    Aggregate cable-level data for Model 2 (Market Demand Forecasting)
    Since we don't have region data, we'll aggregate by risk level and health
    """
    
    print("\n" + "=" * 60)
    print("Creating Model 2 Input Files")
    print("=" * 60)
    
    # Aggregation by Health Index
    health_demand = df.groupby('Health Index').agg({
        'xlpe_demand_tons': 'sum',
        'ID': 'count',
        'Age': 'mean',
        'Partial Discharge': 'mean',
        'Neutral Corrosion': 'mean'
    }).reset_index()
    health_demand.columns = ['health_index', 'total_xlpe_tons', 'cable_count', 
                              'avg_age', 'avg_pd', 'avg_corrosion']
    
    # Aggregation by Risk Level
    risk_demand = df.groupby('risk_level').agg({
        'xlpe_demand_tons': 'sum',
        'ID': 'count',
        'replacement_urgency_years': 'mean'
    }).reset_index()
    risk_demand.columns = ['risk_level', 'total_xlpe_tons', 'cable_count', 'avg_urgency']
    
    # Aggregation by Urgency Band
    df['urgency_band'] = pd.cut(
        df['replacement_urgency_years'],
        bins=[0, 3, 6, 10, 15],
        labels=['Immediate (0-3yr)', 'Short-term (3-6yr)', 
                'Medium-term (6-10yr)', 'Long-term (10-15yr)']
    )
    
    urgency_demand = df.groupby('urgency_band', observed=True).agg({
        'xlpe_demand_tons': 'sum',
        'ID': 'count'
    }).reset_index()
    urgency_demand.columns = ['urgency_band', 'xlpe_demand_tons', 'cable_count']
    
    # Save files
    health_demand.to_csv('model2_health_demand.csv', index=False)
    risk_demand.to_csv('model2_risk_demand.csv', index=False)
    urgency_demand.to_csv('model2_urgency_demand.csv', index=False)
    
    print("✅ Saved: model2_health_demand.csv")
    print("✅ Saved: model2_risk_demand.csv")
    print("✅ Saved: model2_urgency_demand.csv")
    
    print(f"\n📊 Total XLPE Demand: {df['xlpe_demand_tons'].sum():.2f} tons")
    print(f"📊 Cables by Risk Level:")
    print(risk_demand.to_string(index=False))
    
    return health_demand, risk_demand, urgency_demand


if __name__ == "__main__":
    # Load and preprocess real data
    df = load_and_preprocess_real_data('15-KV XLPE Cable.xlsx')
    
    # Save preprocessed data
    df.to_csv('cable_data_processed.csv', index=False)
    print(f"\n✅ Saved preprocessed data to: cable_data_processed.csv")
    
    # Create Model 2 inputs
    create_model2_inputs(df)
    
    print("\n" + "=" * 60)
    print("Data preprocessing complete!")
    print("=" * 60)
    
    print(f"\n✅ Generated {len(df)} inspection records")
    print(f"✅ Saved to: utility_inspection_data.csv")
    
    print("\n📊 Records by Region:")
    print(df['region'].value_counts().to_string())
    
    print("\n📊 Risk Level Distribution:")
    print(df['risk_level'].value_counts().to_string())
    
    print("\n📊 Health Index Stats:")
    print(df['health_index'].describe().round(2).to_string())
    
    print("\n📊 Sample Records:")
    print(df[['inspection_id', 'region', 'cable_age_years', 'health_index', 
              'risk_level', 'replacement_urgency_years', 'xlpe_demand_tons']].head(8).to_string())
