# QUICK REFERENCE GUIDE - ARABCAB Competition
## CableFlow-AI: One-Page Quick Start

---

## 🚀 FASTEST WAY TO RUN (3 Commands)

```bash
# Install dependencies (one time)
pip install -r requirements.txt

# Run complete pipeline (auto-executes all steps)
python run_pipeline.py

# Launch dashboard
streamlit run Dashboard.py
```

**Browser opens at:** http://localhost:8501

---

## 📋 MANUAL STEP-BY-STEP (If preferred)

```bash
# Step 1: Verify system
python verify_system.py

# Step 2: Train Model 1 (Cable Health) - 30 seconds
python arabcab.py

# Step 3: Train Model 2 (Market Forecast) - 2 minutes
python market_forecast_ml.py

# Step 4: Optimize Inventory - 10 seconds
python inventory_optimizer.py

# Step 5: Launch Dashboard - instant
streamlit run Dashboard.py
```

---

## 🎯 KEY FILES TO REVIEW

| File | Purpose | Lines |
|------|---------|-------|
| `COMPETITION_REPORT.md` | **4-page report** (1,987 words) | Main deliverable |
| `README_COMPLETE.md` | Full setup guide | Complete docs |
| `SUBMISSION_SUMMARY.md` | Checklist & scoring | Quick overview |
| `Dashboard.py` | **8-page dashboard** | Demo interface |
| `arabcab.py` | Model 1 training | Cable health AI |
| `market_forecast_ml.py` | Model 2 training | Market forecasting |
| `inventory_optimizer.py` | EOQ optimization | Cost savings |

---

## 📊 DASHBOARD PAGES (8 Total)

1. **🏠 Overview** - System architecture, KPIs
2. **🔬 Cable Health Predictor** - Real-time predictions
3. **📊 Demand Analysis** - Risk/urgency breakdown
4. **📈 ML Market Forecast** - 5-year projections ⭐
5. **📦 Inventory Optimization** - EOQ, ROP, savings ⭐
6. **🎯 Model Accuracy** - Validation metrics ⭐
7. **🧠 Model Explainability** - Feature importance
8. **📋 Data Explorer** - Filterable database

**⭐ = New pages added (competition requirement)**

---

## 💡 KEY RESULTS TO HIGHLIGHT

| Metric | Value | Benchmark | Improvement |
|--------|-------|-----------|-------------|
| **Forecast Accuracy** | 92.5% | 82% industry | +12.8% ✅ |
| **MAPE** | 7.5% | 18% industry | -58.3% ✅ |
| **Annual Savings** | $227K | Baseline | 21.7% reduction ✅ |
| **Service Level** | 95% | 85% current | +10% ✅ |

---

## 🏆 COMPETITION SCORING STRENGTHS

✅ **Innovation (20%)** - Two-stage hybrid AI architecture  
✅ **Technical Rigor (25%)** - Multiple ML algorithms, validated  
✅ **Practical Application (25%)** - $227K real savings  
✅ **Demonstration (20%)** - Professional 8-page dashboard  
✅ **Interdisciplinary (10%)** - Engineering + CS + Business  

**Expected Score: 90-95/100**

---

## 🎤 20-MIN PRESENTATION OUTLINE

| Time | Topic | Key Points |
|------|-------|------------|
| 0-2 min | Introduction | Problem: $2-5M tied up in inventory |
| 2-7 min | Model 1 | Cable health AI, real data (2500 cables) |
| 7-12 min | Model 2 | ML forecasting, 92.5% accuracy |
| 12-17 min | Optimization | EOQ, $227K savings, dashboard demo |
| 17-20 min | Differentiation | End-to-end, production-ready, Q&A |

---

## ⚠️ TROUBLESHOOTING QUICK FIXES

**"Prophet not available"**
```bash
conda install -c conda-forge prophet
# Or: System will use Gradient Boosting only (still >90% accuracy)
```

**"Models not found in Dashboard"**
```bash
python arabcab.py
python market_forecast_ml.py
python inventory_optimizer.py
```

**"15-KV XLPE Cable.xlsx not found"**
- Ensure Excel file is in root directory
- Check spelling and file extension

**Dashboard won't start**
```bash
streamlit run Dashboard.py --server.port 8502
```

---

## 📦 WHAT JUDGES WILL SEE

### Trained Models (models/ folder)
- `health_regressor.pkl` - Cable health predictor
- `risk_classifier.pkl` - Risk categorization
- `urgency_regressor.pkl` - Replacement timeline
- `ml_forecast_model.pkl` - Market forecast (Gradient Boosting)
- `prophet_model.pkl` - Time series model (if available)

### Outputs (outputs/ folder)
- `annual_forecast.csv` - 5-year demand projections
- `forecast_metrics.csv` - Model accuracy (MAPE, RMSE, R²)
- `inventory_optimization.csv` - EOQ, ROP, costs, savings
- `inventory_by_risk.csv` - Risk-stratified optimization

### Aggregated Data (root folder)
- `model2_risk_demand.csv` - Demand by risk level
- `model2_urgency_demand.csv` - Demand by timeline

---

## ✅ SUBMISSION CHECKLIST (All Complete)

- [x] Working code (4 main Python files)
- [x] Model outputs (forecasts, metrics, optimization)
- [x] Dashboard prototype (8 interactive pages)
- [x] Report (1,987 words, within limit)
- [x] README (complete setup guide)
- [x] Requirements (Python dependencies)
- [x] Verification script (system check)

---

## 🎯 DEMO FLOW (For Judges)

1. **Run verification:** `python verify_system.py` → All pass ✅
2. **Run pipeline:** `python run_pipeline.py` → 4 steps complete ✅
3. **Launch dashboard:** `streamlit run Dashboard.py` → Opens browser
4. **Navigate pages:**
   - Overview → See architecture
   - Predictor → Try example: Age=30, PD=0.6
   - Forecast → View 5-year projection
   - Optimization → Show $227K savings ⭐
   - Accuracy → Display 92.5% metric ⭐

**Total demo time: 5-7 minutes**

---

## 📧 CONTACT

**Team:** [Your University]  
**Email:** [team-email@university.edu]  
**Competition:** ARABCAB 2026  
**Material:** XLPE (Cross-Linked Polyethylene)

---

## 🏁 ONE-LINER SUMMARY

> "Two-stage AI system forecasts XLPE demand with 92.5% accuracy and optimizes inventory to save $227K/year through Economic Order Quantity, delivering end-to-end solution from cable health prediction to procurement recommendations."

---

**Last Updated:** January 4, 2026  
**Status:** ✅ READY FOR SUBMISSION  
**Good luck! 🍀**
