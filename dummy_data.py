"""
================================================================================
ARABCAB Competition - Utility Inspection Data Generator
================================================================================
Generates realistic synthetic data representing utility company cable inspections.
This simulates data from: Egypt (EETC), UAE (DEWA, ADDC), Bahrain (EWA)
================================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================
N_RECORDS = 2000

REGIONS = {
    'Egypt': {'weight': 0.50, 'avg_cable_age': 22, 'maintenance_quality': 0.65, 
              'soil_corrosivity': 0.7, 'avg_load_factor': 0.85},
    'UAE': {'weight': 0.30, 'avg_cable_age': 12, 'maintenance_quality': 0.90, 
            'soil_corrosivity': 0.8, 'avg_load_factor': 0.70},
    'Bahrain': {'weight': 0.20, 'avg_cable_age': 18, 'maintenance_quality': 0.80, 
                'soil_corrosivity': 0.85, 'avg_load_factor': 0.75}
}

APPLICATIONS = {
    'Power Transmission (HV)': {'criticality': 0.95},
    'Power Distribution (MV)': {'criticality': 0.80},
    'Industrial': {'criticality': 0.70},
    'Renewable Energy': {'criticality': 0.85},
    'Telecommunications': {'criticality': 0.60},
    'Construction/Building': {'criticality': 0.50},
}

VOLTAGE_CLASSES = ['LV (<1kV)', 'MV (1-35kV)', 'HV (35-150kV)', 'EHV (>150kV)']

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_inspection_date():
    start = datetime(2023, 1, 1)
    end = datetime(2025, 12, 31)
    return start + timedelta(days=np.random.randint(0, (end - start).days))

def generate_cable_record(region_name, region_config):
    cable_age = int(np.clip(np.random.normal(region_config['avg_cable_age'], 
                                              region_config['avg_cable_age'] * 0.4), 1, 50))
    application = np.random.choice(list(APPLICATIONS.keys()))
    voltage_class = np.random.choice(VOLTAGE_CLASSES, p=[0.30, 0.40, 0.20, 0.10])
    
    # Cable length varies by application
    if 'Transmission' in application:
        cable_length = np.random.uniform(5, 50)
    elif 'Distribution' in application:
        cable_length = np.random.uniform(0.5, 10)
    else:
        cable_length = np.random.uniform(0.1, 5)
    
    # Environmental & operational factors
    soil_corrosivity = np.clip(np.random.normal(region_config['soil_corrosivity'], 0.1), 0, 1)
    ambient_temp = np.random.normal(35 if region_name != 'Egypt' else 30, 5)
    humidity = np.random.normal(60 if region_name == 'Bahrain' else 45, 10)
    load_factor = np.clip(np.random.normal(region_config['avg_load_factor'], 0.1), 0.3, 1.0)
    overload_events = np.random.poisson(2 if load_factor > 0.8 else 0.5)
    fault_history = np.random.poisson(cable_age / 10)
    maintenance_score = np.clip(np.random.normal(region_config['maintenance_quality'], 0.15), 0, 1)
    last_maintenance_months = np.random.randint(1, 36)
    
    # Diagnostic measurements (key indicators)
    base_pd = 5 + (cable_age * 0.8) + (fault_history * 3) - (maintenance_score * 10)
    partial_discharge_pC = max(0, np.random.normal(base_pd, 5))
    base_ir = 5000 - (cable_age * 80) - (soil_corrosivity * 500) + (maintenance_score * 1000)
    insulation_resistance = max(100, np.random.normal(base_ir, 500))
    tan_delta = max(0.0005, np.random.normal(0.001 + (cable_age * 0.0002) + (fault_history * 0.001), 0.001))
    
    return {
        'inspection_date': generate_inspection_date(),
        'region': region_name,
        'application': application,
        'voltage_class': voltage_class,
        'installation_year': 2025 - cable_age,
        'cable_age_years': cable_age,
        'cable_length_km': round(cable_length, 2),
        'rated_voltage_kV': {'LV (<1kV)': 0.4, 'MV (1-35kV)': 11, 'HV (35-150kV)': 66, 'EHV (>150kV)': 220}[voltage_class],
        'soil_corrosivity_index': round(soil_corrosivity, 3),
        'ambient_temp_C': round(ambient_temp, 1),
        'humidity_percent': round(humidity, 1),
        'load_factor': round(load_factor, 3),
        'overload_events_last_year': overload_events,
        'fault_history_count': fault_history,
        'maintenance_score': round(maintenance_score, 3),
        'months_since_maintenance': last_maintenance_months,
        'partial_discharge_pC': round(partial_discharge_pC, 2),
        'insulation_resistance_MOhm': round(insulation_resistance, 0),
        'tan_delta': round(tan_delta, 5),
    }

def calculate_health_metrics(df):
    """Calculate target variables for Model 1"""
    
    # HEALTH INDEX (0-100)
    df['health_index'] = (
        100 - (df['cable_age_years'] * 1.2) - (df['partial_discharge_pC'] * 0.5)
        - (df['fault_history_count'] * 5) - ((1 - df['maintenance_score']) * 15)
        - (df['overload_events_last_year'] * 3) + (df['insulation_resistance_MOhm'] / 200)
        - (df['tan_delta'] * 500) - (df['soil_corrosivity_index'] * 10)
    ).clip(0, 100).round(1)
    
    # RISK LEVEL (using apply instead of np.select for compatibility)
    def get_risk_level(health):
        if health >= 70:
            return 'Low'
        elif health >= 50:
            return 'Medium'
        elif health >= 30:
            return 'High'
        else:
            return 'Critical'
    
    df['risk_level'] = df['health_index'].apply(get_risk_level)
    
    # REPLACEMENT URGENCY (years)
    criticality_map = {app: cfg['criticality'] for app, cfg in APPLICATIONS.items()}
    df['criticality_factor'] = df['application'].map(criticality_map)
    base_urgency = (df['health_index'] / 100) * 15
    df['replacement_urgency_years'] = (base_urgency * (2 - df['criticality_factor'])).clip(0.5, 15).round(1)
    
    # XLPE DEMAND (tons) - feeds into Model 2
    voltage_mult = {'LV (<1kV)': 0.8, 'MV (1-35kV)': 1.5, 'HV (35-150kV)': 3.0, 'EHV (>150kV)': 5.5}
    df['voltage_multiplier'] = df['voltage_class'].map(voltage_mult)
    urgency_factor = (15 - df['replacement_urgency_years']) / 15
    df['xlpe_demand_tons'] = (df['cable_length_km'] * df['voltage_multiplier'] * 0.5 * (1 + urgency_factor)).round(2)
    
    df = df.drop(columns=['criticality_factor', 'voltage_multiplier'])
    return df

def generate_dataset():
    records = []
    for region_name, region_config in REGIONS.items():
        for _ in range(int(N_RECORDS * region_config['weight'])):
            records.append(generate_cable_record(region_name, region_config))
    
    df = pd.DataFrame(records)
    df['inspection_id'] = [f"INS-{i:05d}" for i in range(1, len(df) + 1)]
    df = calculate_health_metrics(df)
    
    # Reorder columns
    cols = ['inspection_id', 'inspection_date', 'region', 'application', 'voltage_class',
            'installation_year', 'cable_age_years', 'cable_length_km', 'rated_voltage_kV',
            'soil_corrosivity_index', 'ambient_temp_C', 'humidity_percent', 'load_factor',
            'overload_events_last_year', 'fault_history_count', 'maintenance_score',
            'months_since_maintenance', 'partial_discharge_pC', 'insulation_resistance_MOhm',
            'tan_delta', 'health_index', 'risk_level', 'replacement_urgency_years', 'xlpe_demand_tons']
    return df[cols]

# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("ARABCAB - Utility Inspection Data Generator")
    print("=" * 60)
    
    df = generate_dataset()
    df.to_csv('utility_inspection_data.csv', index=False)
    
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
