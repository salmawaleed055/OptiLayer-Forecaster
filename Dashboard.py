"""
================================================================================
ARABCAB Competition - CableFlow AI Dashboard
================================================================================
Interactive Decision Support System for Cable Health & XLPE Demand Forecasting
Works with REAL 15-KV XLPE Cable inspection data (2500 records)
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
        feature_names = joblib.load('models/feature_names.pkl')
        coefficients = pd.read_csv('models/health_coefficients.csv')
        return {
            'health': (health_model, health_scaler),
            'risk': (risk_model, risk_scaler, risk_encoder),
            'urgency': (urgency_model, urgency_scaler),
            'feature_names': feature_names,
            'coefficients': coefficients,
            'loaded': True
        }
    except Exception as e:
        st.warning(f"⚠️ Models not found. Run `python arabcab.py` first. Error: {e}")
        return {'loaded': False}

@st.cache_data
def load_data():
    """Load real cable data and aggregated demand"""
    try:
        # Load real cable data
        cable_df = pd.read_excel('15-KV XLPE Cable.xlsx')
        cable_df.columns = cable_df.columns.str.strip()
        
        # Add derived columns
        def get_risk_level(health):
            if health >= 5: return 'Low'
            elif health >= 4: return 'Medium'
            elif health >= 3: return 'High'
            else: return 'Critical'
        
        cable_df['risk_level'] = cable_df['Health Index'].apply(get_risk_level)
        cable_df['replacement_urgency_years'] = ((cable_df['Health Index'] / 5) * 15).clip(0.5, 15).round(1)
        
        urgency_factor = (15 - cable_df['replacement_urgency_years']) / 15
        cable_df['xlpe_demand_tons'] = (2.0 * 1.5 * 0.5 * (1 + urgency_factor)).round(2)
        
        # Load aggregated demand files
        try:
            risk_demand = pd.read_csv('model2_risk_demand.csv')
            urgency_demand = pd.read_csv('model2_urgency_demand.csv')
        except:
            risk_demand = None
            urgency_demand = None
            
        try:
            forecast_df = pd.read_csv("outputs/model2_adjusted_forecast.csv")
        except:
            forecast_df = None

        return cable_df, risk_demand, urgency_demand, forecast_df
            

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

models = load_models()
cable_df, risk_demand, urgency_demand, forecast_df = load_data()


# --- HEADER ---
st.title("⚡ CableFlow-AI: 15-KV XLPE Cable Health Dashboard")
st.markdown("**ARABCAB Competition** | AI-Based Demand Forecasting for Cable Industry")
st.divider()

# --- SIDEBAR ---
st.sidebar.header("🎛️ Navigation")
page = st.sidebar.radio("Select View", [
    "🏠 Overview",
    "🔬 Cable Health Predictor",
    "📊 Demand Analysis",
    "📈 Market Forecast (Model 2)", 
    "🧠 Model Explainability",
    "📋 Data Explorer"
])


# ============================================================================
# PAGE 1: OVERVIEW
# ============================================================================
if page == "🏠 Overview":
    st.header("System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if cable_df is not None:
        critical_count = len(cable_df[cable_df['risk_level'] == 'Critical'])
        total_demand = cable_df['xlpe_demand_tons'].sum()
        
        col1.metric("📋 Total Cables", f"{len(cable_df):,}")
        col2.metric("📅 Avg Cable Age", f"{cable_df['Age'].mean():.1f} years")
        col3.metric("⚠️ Critical Cables", f"{critical_count}")
        col4.metric("📦 Total XLPE Demand", f"{total_demand:,.0f} Tons")
    
    st.divider()
    
    # Two-Model Architecture
    st.subheader("🏗️ Two-Stage AI Architecture")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.info("""
        **MODEL 1: Cable Health Prediction**
        
        *Input Features (from real data):*
        - Age (years)
        - Partial Discharge (0-1)
        - Visual Condition (Good/Medium/Poor)
        - Neutral Corrosion (0-1)
        - Loading
        
        *Outputs:*
        - Health Index (1-5)
        - Risk Level (Low/Medium/High/Critical)
        - Replacement Urgency (years)
        
        *Method:* Explainable Ridge Regression
        """)
    
    with col_right:
        st.success("""
        **MODEL 2: Market Demand Forecasting** *(Next Phase)*
        
        *Input:* Aggregated demand from Model 1
        
        *Outputs:*
        - XLPE market size by risk level
        - Demand by urgency timeline
        - Replacement prioritization
        
        *Purpose:* Strategic procurement planning
        """)
    
    # Health Index Distribution
    if cable_df is not None:
        st.subheader("📈 Cable Health Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            health_counts = cable_df['Health Index'].value_counts().sort_index()
            fig_health = px.bar(x=health_counts.index, y=health_counts.values,
                               labels={'x': 'Health Index (1-5)', 'y': 'Number of Cables'},
                               title="Distribution of Health Index",
                               color=health_counts.index,
                               color_continuous_scale='RdYlGn')
            st.plotly_chart(fig_health, use_container_width=True)
        
        with col2:
            risk_counts = cable_df['risk_level'].value_counts()
            fig_risk = px.pie(values=risk_counts.values, names=risk_counts.index,
                             title="Risk Level Distribution",
                             color=risk_counts.index,
                             color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 
                                                'High': '#e74c3c', 'Critical': '#8e44ad'})
            st.plotly_chart(fig_risk, use_container_width=True)

# ============================================================================
# PAGE 2: CABLE HEALTH PREDICTOR
# ============================================================================
elif page == "🔬 Cable Health Predictor":
    st.header("🔬 Real-Time Cable Health Assessment")
    st.markdown("Enter cable inspection data to predict health, risk, and replacement urgency.")
    
    if not models.get('loaded'):
        st.error("❌ Models not loaded. Please run `python arabcab.py` first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📋 Cable Inspection Data")
            age = st.slider("Cable Age (Years)", 1, 60, 25)
            partial_discharge = st.slider("Partial Discharge", 0.0, 1.0, 0.3, 0.01,
                                         help="Normalized PD measurement (0=best, 1=worst)")
            visual_condition = st.selectbox("Visual Condition", ["Good", "Medium", "Poor"])
            neutral_corrosion = st.slider("Neutral Corrosion Index", 0.0, 1.0, 0.5, 0.01,
                                          help="Corrosion level (0=none, 1=severe)")
            loading = st.number_input("Loading", 50, 1000, 400,
                                      help="Cable loading value")
        
        with col2:
            st.subheader("📊 Feature Summary")
            st.write(f"**Age:** {age} years")
            st.write(f"**Partial Discharge:** {partial_discharge}")
            st.write(f"**Visual Condition:** {visual_condition}")
            st.write(f"**Neutral Corrosion:** {neutral_corrosion}")
            st.write(f"**Loading:** {loading}")
        
        st.divider()
        
        if st.button("🔮 Predict Cable Health", type="primary", use_container_width=True):
            # Prepare input data matching the model's expected features
            input_data = {
                'Age': age,
                'Partial Discharge': partial_discharge,
                'Neutral Corrosion': neutral_corrosion,
                'Loading': loading,
                'Visual Condition_Medium': 1 if visual_condition == 'Medium' else 0,
                'Visual Condition_Poor': 1 if visual_condition == 'Poor' else 0,
            }
            
            # Ensure columns match what model expects
            feature_names = models['feature_names']
            input_df = pd.DataFrame([{f: input_data.get(f, 0) for f in feature_names}])
            
            # Predict Health Index
            health_model, health_scaler = models['health']
            health_input = health_scaler.transform(input_df)
            health_index = float(np.clip(health_model.predict(health_input)[0], 1, 5))
            
            # Predict Risk Level
            risk_model, risk_scaler, risk_encoder = models['risk']
            risk_input = risk_scaler.transform(input_df)
            risk_pred = risk_model.predict(risk_input)[0]
            risk_level = risk_encoder.inverse_transform([risk_pred])[0]
            
            # Predict Urgency
            urgency_model, urgency_scaler = models['urgency']
            urgency_input = urgency_scaler.transform(input_df)
            urgency_years = float(np.clip(urgency_model.predict(urgency_input)[0], 0.5, 15))
            
            # Calculate XLPE demand
            urgency_factor = (15 - urgency_years) / 15
            xlpe_demand = 2.0 * 1.5 * 0.5 * (1 + urgency_factor)
            
            # Display Results
            st.divider()
            st.subheader("📊 Prediction Results")
            
            res_col1, res_col2, res_col3 = st.columns(3)
            
            with res_col1:
                # Health Index Gauge (1-5 scale)
                fig_health = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=health_index,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Health Index (1-5)"},
                    number={'valueformat': '.2f'},
                    gauge={
                        'axis': {'range': [1, 5]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [1, 2], 'color': "#e74c3c"},
                            {'range': [2, 3], 'color': "#f39c12"},
                            {'range': [3, 4], 'color': "#f1c40f"},
                            {'range': [4, 5], 'color': "#2ecc71"}
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
                st.metric("Est. XLPE Demand", f"{xlpe_demand:.2f} Tons")
            
            with res_col3:
                if risk_level == 'Critical':
                    st.error("🚨 **IMMEDIATE ACTION REQUIRED**\nSchedule replacement within 6 months.")
                elif risk_level == 'High':
                    st.warning("⚠️ **HIGH PRIORITY**\nPlan replacement within 1-2 years.")
                elif risk_level == 'Medium':
                    st.info("📋 **MONITOR CLOSELY**\nSchedule detailed inspection. Plan for 3-5 years.")
                else:
                    st.success("✅ **HEALTHY**\nRoutine monitoring sufficient.")

# ============================================================================
# PAGE 3: DEMAND ANALYSIS
# ============================================================================
elif page == "📊 Demand Analysis":
    st.header("📊 XLPE Demand Analysis")
    st.markdown("Aggregated demand data from Model 1 → Input for Model 2 (Market Forecasting)")
    
    if cable_df is None:
        st.error("❌ Data not loaded.")
    else:
        # Risk Level Demand
        st.subheader("⚠️ XLPE Demand by Risk Level")
        
        if risk_demand is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(risk_demand.round(2), use_container_width=True, hide_index=True)
            
            with col2:
                fig = px.bar(risk_demand, x='risk_level', y='total_xlpe_tons',
                            color='risk_level',
                            color_discrete_map={'Low': '#2ecc71', 'Medium': '#f39c12', 
                                               'High': '#e74c3c', 'Critical': '#8e44ad'},
                            title="XLPE Demand by Risk Level",
                            labels={'total_xlpe_tons': 'XLPE Demand (Tons)', 'risk_level': 'Risk Level'})
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Urgency-Based Demand
        st.subheader("⏰ Demand by Replacement Urgency")
        
        if urgency_demand is not None:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(urgency_demand.round(2), use_container_width=True, hide_index=True)
            
            with col2:
                fig_urgency = px.bar(urgency_demand, x='urgency_band', y='xlpe_demand_tons',
                                    title="XLPE Demand by Urgency Timeline",
                                    labels={'xlpe_demand_tons': 'XLPE Demand (Tons)', 
                                           'urgency_band': 'Replacement Timeline'},
                                    color='xlpe_demand_tons',
                                    color_continuous_scale='Reds')
                st.plotly_chart(fig_urgency, use_container_width=True)
        
        st.divider()
        
        # Summary Stats
        st.subheader("📈 Summary Statistics")
        total_demand = cable_df['xlpe_demand_tons'].sum()
        critical_demand = cable_df[cable_df['risk_level'] == 'Critical']['xlpe_demand_tons'].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total XLPE Demand", f"{total_demand:,.0f} Tons")
        col2.metric("Critical Cable Demand", f"{critical_demand:,.0f} Tons")
        col3.metric("% Critical", f"{(critical_demand/total_demand)*100:.1f}%")

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
        
        # Feature coefficient chart
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
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Coefficient table
        st.subheader("📋 Coefficient Details")
        st.dataframe(coef_df.round(4), use_container_width=True, hide_index=True)
        
        # Business Insights
        st.divider()
        st.subheader("💡 Business Insights from Model")
        
        st.info("""
        **Key Findings for 15-KV Cable Asset Management:**
        
        Based on the model coefficients, the factors affecting cable health are:
        
        1. **Partial Discharge** - Higher PD readings indicate insulation degradation
        2. **Neutral Corrosion** - Corrosion directly damages cable integrity
        3. **Age** - Older cables naturally have lower health indices
        4. **Visual Condition** - Physical inspection results correlate with health
        5. **Loading** - Higher loads can accelerate cable wear
        
        **Actionable Recommendations:**
        - Prioritize replacement for cables with high PD (>0.5) and corrosion (>0.7)
        - Focus on cables >30 years old with Poor visual condition
        - Regular monitoring of high-load cables (>500)
        """)

# ============================================================================
# PAGE 5: DATA EXPLORER
# ============================================================================
elif page == "📋 Data Explorer":
    st.header("📋 Cable Data Explorer")
    
    if cable_df is None:
        st.error("❌ Data not loaded.")
    else:
        st.subheader("🔍 Filter Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            age_range = st.slider("Age Range", 
                                  int(cable_df['Age'].min()), 
                                  int(cable_df['Age'].max()),
                                  (int(cable_df['Age'].min()), int(cable_df['Age'].max())))
        
        with col2:
            health_filter = st.multiselect("Health Index", 
                                           options=sorted(cable_df['Health Index'].unique()),
                                           default=sorted(cable_df['Health Index'].unique()))
        
        with col3:
            risk_filter = st.multiselect("Risk Level",
                                         options=['Low', 'Medium', 'High', 'Critical'],
                                         default=['Low', 'Medium', 'High', 'Critical'])
        
        # Filter data
        filtered_df = cable_df[
            (cable_df['Age'] >= age_range[0]) & 
            (cable_df['Age'] <= age_range[1]) &
            (cable_df['Health Index'].isin(health_filter)) &
            (cable_df['risk_level'].isin(risk_filter))
        ]
        
        st.write(f"**Showing {len(filtered_df)} of {len(cable_df)} cables**")
        
        # Display data
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data",
            data=csv,
            file_name="filtered_cable_data.csv",
            mime="text/csv"
        )
elif page == "📈 Market Forecast (Model 2)":
    st.header("📈 XLPE Market Forecast (Model 2)")
    st.markdown("Hybrid rule-based baseline + ML-inspired trend adjustment")

    if forecast_df is None:
        st.error("❌ Run `python model2_market_forecast.py` first.")
    else:
        # -----------------------------
        # SLIDERS (USER-CONTROLLED ASSUMPTIONS)
        # -----------------------------
        st.subheader("🎛️ Market Assumptions")

        col1, col2, col3 = st.columns(3)

        with col1:
            pe_price_index = st.slider(
                "Polyethylene Price Index",
                80, 140, 110,
                help="Relative price index (100 = historical average)"
            )

        with col2:
            grid_growth = st.slider(
                "Grid Expansion Rate (%)",
                1.0, 10.0, 5.0,
                step=0.5,
                help="Annual power grid expansion rate"
            ) / 100

        with col3:
            supply_delay = st.slider(
                "Supply Chain Delay (weeks)",
                0, 12, 2,
                help="Average delivery delay"
            )

        # -----------------------------
        # TREND FUNCTION (EXPLAINABLE)
        # -----------------------------
        def compute_trend_factor(price, growth, delay):
            trend = 0

            if price > 100:
                trend -= 0.05
            else:
                trend += 0.03

            if growth > 0.04:
                trend += 0.07
            else:
                trend -= 0.02

            if delay > 4:
                trend -= 0.04

            return trend

        trend_factor = compute_trend_factor(
            pe_price_index,
            grid_growth,
            supply_delay
        )

        # -----------------------------
        # APPLY LIVE ADJUSTMENT
        # -----------------------------
        adjusted_df = forecast_df.copy()
        adjusted_df["Live_Adjusted_Demand"] = (
            adjusted_df["Baseline_XLPE_Demand_Tons"]
            * (1 + trend_factor)
        ).round(2)

        # -----------------------------
        # KPIs
        # -----------------------------
        st.divider()
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Trend Adjustment",
            f"{trend_factor*100:+.1f}%"
        )

        col2.metric(
            "Peak Demand (Tons)",
            f"{adjusted_df['Live_Adjusted_Demand'].max():,.0f}"
        )

        col3.metric(
            "Total Forecast Demand",
            f"{adjusted_df['Live_Adjusted_Demand'].sum():,.0f} Tons"
        )

        # -----------------------------
        # LINE CHART
        # -----------------------------
        st.subheader("📈 XLPE Market Demand Forecast")

        fig = px.line(
            adjusted_df,
            x="Year",
            y="Live_Adjusted_Demand",
            markers=True,
            title="XLPE Market Demand Forecast (Live Scenario)",
            labels={
                "Live_Adjusted_Demand": "XLPE Demand (Tons)"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # DATA TABLE
        # -----------------------------
        with st.expander("📋 View Forecast Table"):
            st.dataframe(
                adjusted_df[[
                    "Year",
                    "Baseline_XLPE_Demand_Tons",
                    "Live_Adjusted_Demand"
                ]],
                use_container_width=True
            )


# --- FOOTER ---
st.divider()
st.caption("⚡ CableFlow AI | ARABCAB Scientific Competition 2026 | Using Real 15-KV XLPE Cable Data")