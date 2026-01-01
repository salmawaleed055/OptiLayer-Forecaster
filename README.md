# CableFlow-AI: XLPE Demand Forecasting & Inventory Optimization


## Project Overview

### Key Problems Addressed:
* [cite_start]**Demand Fluctuations:** Unpredictable shifts due to changing infrastructure projects and renewable energy expansion[cite: 8, 31].
* [cite_start]**Price Volatility:** High sensitivity to global commodity price trends in polymers[cite: 10, 11].
* [cite_start]**Inventory Inefficiency:** Eliminating the risks of overstocking capital and understocking raw materials[cite: 14, 15].

## Technical Stack
* **Language:** Python 3.13
* **AI Model:** XGBoost Regressor (Extreme Gradient Boosting)
* [cite_start]**Dashboard:** Streamlit (Interactive Visualization) [cite: 61, 186]
* **Libraries:** Pandas, Scikit-learn, Plotly, Joblib

##  Features
- [cite_start]**AI-Driven Forecasting:** Uses historical data, infrastructure age, and market price lags to predict next-month XLPE demand[cite: 19, 20].
- [cite_start]**Inventory Optimization Tool:** Automatically calculates optimal order quantities and safety stock levels based on AI predictions[cite: 21, 72].
- [cite_start]**Live Decision Dashboard:** An interactive interface for procurement managers to simulate different market scenarios[cite: 22, 94].

##  Project Structure
- `arabcab.py`: The "Engine Room" where the XGBoost model is trained and saved.
- `Dashboard.py`: The Streamlit application providing the interactive UI.
- `models/`: Contains the serialized AI brain (`.pkl` files).
- [cite_start]`utility_cable_data.csv`: Validated sample data used for the Proof of Concept[cite: 73, 188].



---
