import pandas as pd
import numpy as np

# Create 100 rows of fake cable data
data = {
    'cable_age': np.random.randint(1, 50, 100),
    'electricity_usage': np.random.randint(500, 3000, 100),
    'pe_price_lag_30': np.random.randint(80, 150, 100),
    'lead_time_days': np.random.randint(7, 90, 100),
    'xlpe_demand_tons': np.random.randint(50, 250, 100)
}

df = pd.DataFrame(data)

# Save it to the exact name your arabcab.py expects
df.to_csv('utility_cable_data.csv', index=False)
print("✅ Created utility_cable_data.csv successfully!")
