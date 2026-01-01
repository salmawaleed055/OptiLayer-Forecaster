# CableFlow-AI: XLPE Demand Forecasting & Inventory Optimization


## Project Overview

### Key Problems Addressed:
* **Demand Fluctuations:** Unpredictable shifts due to changing infrastructure projects and renewable energy expansion[cite: 8, 31].
* **Price Volatility:** High sensitivity to global commodity price trends in polymers.
* **Inventory Inefficiency:** Eliminating the risks of overstocking capital and understocking raw materials.

## Technical Stack
* **Language:** Python 3.13
* **AI Model:** XGBoost Regressor (Extreme Gradient Boosting)
* **Dashboard:** Streamlit (Interactive Visualization) 
* **Libraries:** Pandas, Scikit-learn, Plotly, Joblib

##  Features
- **AI-Driven Forecasting:** Uses historical data, infrastructure age, and market price lags to predict next-month XLPE demand.
- **Inventory Optimization Tool:** Automatically calculates optimal order quantities and safety stock levels based on AI predictions.
- **Live Decision Dashboard:** An interactive interface for procurement managers to simulate different market scenarios.

##  Project Structure
- `arabcab.py`: The "Engine Room" where the XGBoost model is trained and saved.
- `Dashboard.py`: The Streamlit application providing the interactive UI.
- `models/`: Contains the serialized AI brain (`.pkl` files).
- `utility_cable_data.csv`: Validated sample data used for the Proof of Concept.



---
