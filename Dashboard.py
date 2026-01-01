"""
================================================================================
ARABCAB Competition - CableFlow AI Dashboard
================================================================================
Interactive Decision Support System for Cable Health & XLPE Demand Forecasting
Integrates with Model 1 (Health Prediction) outputs
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="CableFlow AI | Arabcab", layout="wide", page_icon="⚡")

# --- MODEL LOADING ---
@st.cache_resource
def load_models():
    """Load all Model 1 components"""
    try:
        health_model = joblib.load('models/health_regressor.pkl')
        health_scaler = joblib.load('models/health_scaler.pkl')
        risk_model = joblib.load('models/risk_classifier.pkl')
        risk_scaler = joblib.load('models/risk_scaler.pkl')
        risk_encoder = joblib.load('models/risk_encoder.pkl')
        urgency_model = joblib.load('models/urgency_regressor.pkl')
        urgency_scaler = joblib.load('models/urgency_scaler.pkl')
        coefficients = pd.read_csv('models/health_coefficients.csv')
        return {
            'health': (health_model, health_scaler),
            'risk': (risk_model, risk_scaler, risk_encoder),
            'urgency': (urgency_model, urgency_scaler),
            'coefficients': coefficients,
            'loaded': True
        }
    except Exception as e:
        st.warning(f"⚠️ Models not found. Run `python arabcab.py` first. Error: {e}")
        return {'loaded': False}

@st.cache_data
def load_data():
    """Load inspection data and aggregated demand"""
    try:
        inspection_df = pd.read_csv('utility_inspection_data.csv')
        regional_demand = pd.read_csv('model2_regional_demand.csv')
        urgency_demand = pd.read_csv('model2_urgency_demand.csv')
        return inspection_df, regional_demand, urgency_demand
    except:
        return None, None, None

models = load_models()
inspection_df, regional_demand, urgency_demand = load_data()

# --- HEADER ---
st.title("⚡ CableFlow-AI: Cable Health & Demand Dashboard")
st.markdown("**ARABCAB Competition** | AI-Based Demand Forecasting for Cable Industry")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("🎛️ Navigation")
page = st.sidebar.radio("Select View", [
    "🏠 Overview",
    "🔬 Cable Health Predictor", 
    "📊 Regional Demand Analysis",
    "🧠 Model Explainability"
])

# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.header("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if inspection_df is not None:
        col1.metric("📋 Total Inspections", f"{len(inspection_df):,}")
        col2.metric("🌍 Regions Covered", inspection_df['region'].nunique())
        col3.metric("⚠️ Critical Cables", len(inspection_df[inspection_df['risk_level'] == 'Critical']))
        col4.metric("📦 Total XLPE Demand", f"{inspection_df['xlpe_demand_tons'].sum():,.0f} Tons")
    
    st.divider()
    
    # Two-Model Architecture Diagram
    st.subheader("🏗️ Two-Stage AI Architecture")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.info("""
        **MODEL 1: Utility Health Prediction**
        
        *Input:* Cable inspection data (age, diagnostics, environment)
        
        *Outputs:*
        - Health Index (0-100)
        - Risk Level (Low/Medium/High/Critical)
        - Replacement Urgency (years)
        - Estimated XLPE Demand (tons)
        
        *Method:* Explainable Linear Regression
        """)
    
    with col_right:
        st.success("""
        **MODEL 2: Market Demand Forecasting** *(Next Phase)*
        
        *Input:* Aggregated demand from Model 1
        
        *Outputs:*
        - Regional XLPE market size
        - Growth projections
        - Revenue impact analysis
        
        *Purpose:* Strategic business forecasting
        """)
    
    # Risk Distribution Chart
    if inspection_df is not None:
        st.subheader("📈 Cable Risk Distribution by Region")
        
        risk_counts = inspection_df.groupby(['region', 'risk_level']).size().reset_index(name='count')
        fig = px.bar(risk_counts, x='region', y='count', color='risk_level',
                     color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 
                                         'High': '#e74c3c', 'Critical': '#8e44ad'},
                     barmode='group', title="Cable Count by Risk Level per Region")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE 2: CABLE HEALTH PREDICTOR
# ============================================================================
elif page == "🔬 Cable Health Predictor":
    st.header("🔬 Real-Time Cable Health Assessment")
    st.markdown("Enter cable inspection data to predict health, risk, and replacement urgency.")
    
    if not models.get('loaded'):
        st.error("❌ Models not loaded. Please run `python arabcab.py` first.")
    else:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📍 Cable Info")
            region = st.selectbox("Region", ["Egypt", "UAE", "Bahrain"])
            application = st.selectbox("Application", [
                "Power Transmission (HV)", "Power Distribution (MV)", 
                "Industrial", "Renewable Energy", "Telecommunications", "Construction/Building"
            ])
            voltage_class = st.selectbox("Voltage Class", [
                "LV (<1kV)", "MV (1-35kV)", "HV (35-150kV)", "EHV (>150kV)"
            ])
            cable_age = st.slider("Cable Age (Years)", 1, 50, 15)
            cable_length = st.number_input("Cable Length (km)", 0.1, 100.0, 5.0)
        
        with col2:
            st.subheader("🌡️ Environment & Operations")
            soil_corrosivity = st.slider("Soil Corrosivity Index", 0.0, 1.0, 0.7)
            ambient_temp = st.slider("Ambient Temp (°C)", 15, 50, 35)
            humidity = st.slider("Humidity (%)", 20, 90, 50)
            load_factor = st.slider("Load Factor", 0.3, 1.0, 0.75)
            overload_events = st.number_input("Overload Events (last year)", 0, 20, 2)
            fault_history = st.number_input("Fault History Count", 0, 15, 1)
        
        with col3:
            st.subheader("🔧 Maintenance & Diagnostics")
            maintenance_score = st.slider("Maintenance Score", 0.0, 1.0, 0.7)
            months_since_maint = st.slider("Months Since Maintenance", 1, 36, 12)
            partial_discharge = st.number_input("Partial Discharge (pC)", 0.0, 100.0, 15.0)
            insulation_resistance = st.number_input("Insulation Resistance (MΩ)", 100, 6000, 4000)
            tan_delta = st.number_input("Tan Delta", 0.0005, 0.02, 0.002, format="%.4f")
        
        st.divider()
        
        if st.button("🔮 Predict Cable Health", type="primary", use_container_width=True):
            # Prepare input data
            input_data = {
                'cable_age_years': cable_age,
                'cable_length_km': cable_length,
                'soil_corrosivity_index': soil_corrosivity,
                'ambient_temp_C': ambient_temp,
                'humidity_percent': humidity,
                'load_factor': load_factor,
                'overload_events_last_year': overload_events,
                'fault_history_count': fault_history,
                'maintenance_score': maintenance_score,
                'months_since_maintenance': months_since_maint,
                'partial_discharge_pC': partial_discharge,
                'insulation_resistance_MOhm': insulation_resistance,
                'tan_delta': tan_delta,
                # One-hot encoded categoricals
                'region_Egypt': 1 if region == 'Egypt' else 0,
                'region_UAE': 1 if region == 'UAE' else 0,
                'application_Industrial': 1 if application == 'Industrial' else 0,
                'application_Power Distribution (MV)': 1 if application == 'Power Distribution (MV)' else 0,
                'application_Power Transmission (HV)': 1 if application == 'Power Transmission (HV)' else 0,
                'application_Renewable Energy': 1 if application == 'Renewable Energy' else 0,
                'application_Telecommunications': 1 if application == 'Telecommunications' else 0,
                'voltage_class_HV (35-150kV)': 1 if voltage_class == 'HV (35-150kV)' else 0,
                'voltage_class_LV (<1kV)': 1 if voltage_class == 'LV (<1kV)' else 0,
                'voltage_class_MV (1-35kV)': 1 if voltage_class == 'MV (1-35kV)' else 0,
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Predict Health Index
            health_model, health_scaler = models['health']
            health_input = health_scaler.transform(input_df)
            health_index = float(np.clip(health_model.predict(health_input)[0], 0, 100))
            
            # Predict Risk Level
            risk_model, risk_scaler, risk_encoder = models['risk']
            risk_input = risk_scaler.transform(input_df)
            risk_pred = risk_model.predict(risk_input)[0]
            risk_level = risk_encoder.inverse_transform([risk_pred])[0]
            
            # Predict Urgency
            urgency_model, urgency_scaler = models['urgency']
            urgency_input = urgency_scaler.transform(input_df)
            urgency_years = float(np.clip(urgency_model.predict(urgency_input)[0], 0.5, 15))
            
            # Display Results
            st.divider()
            st.subheader("📊 Prediction Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                # Health Index Gauge
                fig_health = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=health_index,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Health Index"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "#e74c3c"},
                            {'range': [30, 50], 'color': "#f39c12"},
                            {'range': [50, 70], 'color': "#f1c40f"},
                            {'range': [70, 100], 'color': "#2ecc71"}
                        ],
                        'threshold': {'line': {'color': "black", 'width': 4}, 
                                      'thickness': 0.75, 'value': health_index}
                    }
                ))
                fig_health.update_layout(height=250)
                st.plotly_chart(fig_health, use_container_width=True)
            
            with res_col2:
                risk_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🟠', 'Critical': '🔴'}
                st.metric("Risk Level", f"{risk_colors.get(risk_level, '')} {risk_level}")
                st.metric("Replacement Urgency", f"{urgency_years:.1f} Years")
                
                # Estimated XLPE demand
                voltage_mult = {'LV (<1kV)': 0.8, 'MV (1-35kV)': 1.5, 'HV (35-150kV)': 3.0, 'EHV (>150kV)': 5.5}
                urgency_factor = (15 - urgency_years) / 15
                xlpe_demand = cable_length * voltage_mult[voltage_class] * 0.5 * (1 + urgency_factor)
                st.metric("Est. XLPE Demand", f"{xlpe_demand:.2f} Tons")
            
            with res_col3:
                # Recommendation
                if risk_level == 'Critical':
                    st.error("🚨 **IMMEDIATE ACTION REQUIRED**\nSchedule replacement within 6 months.")
                elif risk_level == 'High':
                    st.warning("⚠️ **HIGH PRIORITY**\nPlan replacement within 1-2 years.")
                elif risk_level == 'Medium':
                    st.info("📋 **MONITOR CLOSELY**\nSchedule detailed inspection. Plan for 3-5 years.")
                else:
                    st.success("✅ **HEALTHY**\nRoutine monitoring sufficient.")

# ============================================================================
# PAGE 3: REGIONAL DEMAND ANALYSIS
# ============================================================================
elif page == "📊 Regional Demand Analysis":
    st.header("📊 Regional XLPE Demand Analysis")
    st.markdown("Aggregated demand data from Model 1 → Input for Model 2 (Market Forecasting)")
    
    if regional_demand is None:
        st.error("❌ Run `python arabcab.py` to generate demand data.")
    else:
        # Regional Summary
        st.subheader("🌍 XLPE Demand by Region")
        
        region_summary = regional_demand.groupby('region').agg({
            'total_xlpe_tons': 'sum',
            'cable_count': 'sum',
            'avg_health': 'mean'
        }).reset_index()
        region_summary.columns = ['Region', 'Total XLPE (Tons)', 'Cables Assessed', 'Avg Health Index']
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(region_summary.round(2), use_container_width=True, hide_index=True)
        
        with col2:
            fig = px.pie(region_summary, values='Total XLPE (Tons)', names='Region',
                        title="XLPE Demand Distribution by Region",
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Urgency-Based Demand
        st.subheader("⏰ Demand by Replacement Urgency")
        
        if urgency_demand is not None:
            fig_urgency = px.bar(urgency_demand, x='urgency_band', y='xlpe_demand_tons',
                                 color='region', barmode='group',
                                 title="XLPE Demand by Urgency Band",
                                 labels={'xlpe_demand_tons': 'XLPE Demand (Tons)', 
                                        'urgency_band': 'Replacement Timeline'})
            st.plotly_chart(fig_urgency, use_container_width=True)
            
            # Summary table
            st.dataframe(urgency_demand, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Quarterly Trend
        st.subheader("📅 Quarterly Demand Trend")
        regional_demand['period'] = regional_demand['year'].astype(str) + '-Q' + regional_demand['quarter'].astype(str)
        fig_trend = px.line(regional_demand, x='period', y='total_xlpe_tons', color='region',
                           markers=True, title="XLPE Demand Over Time by Region")
        st.plotly_chart(fig_trend, use_container_width=True)

# ============================================================================
# PAGE 4: MODEL EXPLAINABILITY
# ============================================================================
elif page == "🧠 Model Explainability":
    st.header("🧠 Model Explainability - Why These Predictions?")
    st.markdown("Understanding the linear regression coefficients that drive health predictions.")
    
    if not models.get('loaded'):
        st.error("❌ Models not loaded.")
    else:
        coef_df = models['coefficients']
        
        st.subheader("📊 Feature Impact on Cable Health")
        st.markdown("""
        **Interpretation:**
        - **Positive coefficient** → Feature IMPROVES health index
        - **Negative coefficient** → Feature DEGRADES health index
        - **Magnitude** → Strength of impact (per 1 standard deviation change)
        """)
        
        # Top positive and negative features
        top_positive = coef_df[coef_df['coefficient'] > 0].nlargest(5, 'coefficient')
        top_negative = coef_df[coef_df['coefficient'] < 0].nsmallest(5, 'coefficient')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.success("**✅ Factors that IMPROVE Health:**")
            for _, row in top_positive.iterrows():
                st.write(f"↑ **{row['feature']}**: +{row['coefficient']:.3f}")
        
        with col2:
            st.error("**❌ Factors that DEGRADE Health:**")
            for _, row in top_negative.iterrows():
                st.write(f"↓ **{row['feature']}**: {row['coefficient']:.3f}")
        
        st.divider()
        
        # Full coefficient chart
        st.subheader("📈 All Feature Coefficients")
        
        coef_sorted = coef_df.sort_values('coefficient')
        colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in coef_sorted['coefficient']]
        
        fig = go.Figure(go.Bar(
            x=coef_sorted['coefficient'],
            y=coef_sorted['feature'],
            orientation='h',
            marker_color=colors
        ))
        fig.update_layout(
            title="Health Index Model Coefficients",
            xaxis_title="Coefficient Value",
            yaxis_title="Feature",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Business Insights
        st.divider()
        st.subheader("💡 Business Insights from Model")
        
        st.info("""
        **Key Findings for Cable Asset Management:**
        
        1. **Cable Age** is the strongest predictor of health degradation (-9.47 points per std dev)
        2. **Fault History** significantly impacts health - each historical fault matters (-7.08)
        3. **Partial Discharge** readings are critical diagnostic indicators (-5.61)
        4. **Insulation Resistance** is the best positive indicator (+3.85)
        5. **Maintenance Quality** directly improves cable lifespan (+2.37)
        
        **Actionable Recommendations:**
        - Prioritize maintenance for cables with high PD readings
        - Focus replacement planning on cables >20 years old with fault history
        - Invest in insulation monitoring for early degradation detection
        """)

# --- FOOTER ---
st.divider()
st.caption("⚡ CableFlow AI | ARABCAB Scientific Competition 2026 | Model 1: Utility Health Prediction")