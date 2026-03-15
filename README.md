<div align="center">

# 🎯 Customer Churn & LTV Prediction Engine

### End-to-End Machine Learning System for Customer Intelligence

[![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Churn_Model-green?style=for-the-badge)](https://lightgbm.readthedocs.io)
[![RandomForest](https://img.shields.io/badge/RandomForest-LTV_Model-orange?style=for-the-badge)](https://scikit-learn.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker)](https://docker.com)

</div>

---
🔗 **Technical Report**

https://htmlpreview.github.io/?https://github.com/Pratikchetry/Customer_Churn_LTV_Engine/blob/main/reports/project_report.html

The report contains the full analysis and visualisations used to design the production system.

---

## 📋 Overview

The **Customer Churn & LTV Prediction Engine** is a
production-grade machine learning system that combines
churn risk prediction and lifetime value forecasting
to deliver actionable customer intelligence.

The system enables businesses to:
- Identify customers at risk of churning
- Predict the revenue impact of customer loss
- Segment customers into behavioural groups
- Simulate retention campaign ROI
- Serve predictions through a REST API
- Visualise insights through an interactive dashboard

The project demonstrates the complete ML engineering
lifecycle — from raw data and feature engineering
through model training, explainability, segmentation,
API deployment and containerisation.

---

## 🏆 Key Results

| Model | Algorithm | Metric | Score |
|-------|-----------|--------|-------|
| Churn | LightGBM | F1 Score | **0.8770** |
| Churn | LightGBM | ROC-AUC | **0.9836** |
| Churn | LightGBM | Recall | **1.0000** |
| LTV | RandomForest | R² Score | **0.9226** |
| LTV | RandomForest | RMSE | **$2,806** |

### Segmentation Results

| Segment | Customers | Avg LTV | Total LTV |
|---------|-----------|---------|-----------|
| 🟢 Champion | 2,472 | $21,231 | $52.5M |
| 🟡 At-Risk VIP | 828 | $11,106 | $9.2M |
| 🔵 Promising | 2,550 | $3,725 | $9.5M |
| 🟠 Vulnerable | 850 | $2,093 | $1.8M |
| ⚪ Hibernating | 2,474 | $547 | $1.4M |
| 🔴 Losing Customer | 826 | $75 | $0.1M |

### Revenue Recovery Simulation

| Segment | Investment | Net ROI | ROI % |
|---------|-----------|---------|-------|
| At-Risk VIP | $41,400 | $4.10M | **9,896%** |
| Vulnerable | $21,250 | $0.60M | **2,830%** |
| Losing Customer | $4,130 | $0.01M | **126%** |
| **Total** | **$66,780** | **$4.70M** | **7,043%** |

---

## 🏗️ Project Architecture
```
Raw Data
    ↓
Feature Engineering
    ↓
┌───────────────────┬──────────────────┐
│  Churn Model      │   LTV Model      │
│  LightGBM         │   RandomForest   │
│  F1=0.8770        │   R²=0.9226      │
│  AUC=0.9836       │   RMSE=$2,806    │
└───────────────────┴──────────────────┘
    ↓
SHAP Explainability
    ↓
Customer Segmentation
(Composite Risk Score +
 Within-Tier Percentile Ranking)
    ↓
┌───────────────────┬──────────────────┐
│  FastAPI          │  Streamlit       │
│  REST API         │  Dashboard       │
│  Port 8000        │  Port 8501       │
└───────────────────┴──────────────────┘
    ↓
Docker Compose Deployment
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12 |
| ML — Churn | LightGBM |
| ML — LTV | Scikit-learn RandomForest |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Containerisation | Docker + Docker Compose |
| Data | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn, Plotly |

---

## 📁 Project Structure
```
Customer_Churn_LTV_Engine/
│
├── data/
│   ├── ecommerce_user_segmentation.csv  ← raw data
│   └── customer_segments.csv            ← output
│
├── models/
│   ├── churn_model.pkl                  ← LightGBM
│   └── ltv_model.pkl                    ← RandomForest
│
├── notebooks/
│   ├── 01_model_comparison.ipynb        ← exploration
│   ├── 02_synthetic_data_experiment.ipynb
│   ├── 03_shap_analysis.ipynb
│   └── 04_customer_segmentation.ipynb
│
├── src/
│   ├── feature_engineering.py           ← features
│   ├── train_churn_model.py             ← churn model
│   ├── train_ltv_model.py               ← LTV model
│   ├── model_comparision.py             ← benchmarks
│   ├── customer_segmentation.py         ← segments
│   ├── predict.py                       ← prediction
│   └── pipeline.py                      ← full pipeline
│
├── api/
│   └── main.py                          ← FastAPI
│
├── app/
│   └── streamlit_app.py                 ← dashboard
│
├── reports/
│   ├── shap_churn_summary.png
│   ├── shap_churn_bar.png
│   ├── shap_churn_dependence.png
│   ├── shap_ltv_summary.png
│   ├── shap_ltv_bar.png
│   ├── shap_ltv_dependence.png
│   ├── shap_cross_model.png
│   ├── customer_quadrant.png
│   ├── segment_analysis.png
│   └── revenue_recovery.png
│
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.app
│
├── .streamlit/
│   └── config.toml
│
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### Option 1 — Local Development
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Customer_Churn_LTV_Engine.git
cd Customer_Churn_LTV_Engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run full pipeline
cd src && python pipeline.py

# Start API
cd ../api && uvicorn main:app --reload

# Start dashboard (new terminal)
cd ../app && streamlit run streamlit_app.py
```

### Option 2 — Docker Compose
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/Customer_Churn_LTV_Engine.git
cd Customer_Churn_LTV_Engine

# Build and run
docker compose up --build

# Access:
# API       → http://localhost:8000
# API Docs  → http://localhost:8000/docs
# Dashboard → http://localhost:8501
```

---

## 🚀 Running Each Component

### Full Pipeline
```bash
cd src && python pipeline.py
```

Runs all 5 steps in sequence:
```
[1/5] Feature Engineering    0.5s  ✅
[2/5] Train Churn Model       5.1s  ✅
[3/5] Train LTV Model         7.6s  ✅
[4/5] Model Comparison        4.5s  ✅
[5/5] Customer Segmentation   1.4s  ✅
─────────────────────────────────────
Total Runtime                19.1s  ✅
```

### Single Customer Prediction
```bash
# By customer ID
cd src && python predict.py --customer_id CUST00001

# Random customer
cd src && python predict.py --random
```

Example output:
```
═══════════════════════════════════════════
   CUSTOMER PREDICTION REPORT
═══════════════════════════════════════════
  Customer ID        : CUST00001
  ─────────────────────────────────────────
  Churn Probability  : 0.0000
  Churn Risk         : 🟢 LOW
  ─────────────────────────────────────────
  Predicted LTV      : $3,340.30
  LTV Tier           : Mid LTV
  ─────────────────────────────────────────
  Composite Risk     : 0.1900
  Segment            : 🔵 Promising
  ─────────────────────────────────────────
  RECOMMENDED ACTIONS:
    → Upsell to higher order value products
    → Frequency incentives — buy 3 get 1
    → Wishlist-based recommendations
    → Convert to Champion tier focus
═══════════════════════════════════════════
```

---

## 🌐 API Documentation

The REST API is built with FastAPI and provides
four endpoints:

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health check |
| GET | `/predict/{customer_id}` | Predict by ID |
| POST | `/predict` | Predict by features |
| GET | `/segment/summary` | Segment summary |

### Example — Predict by Customer ID
```bash
curl http://localhost:8000/predict/CUST00001
```

Response:
```json
{
  "customer_id": "CUST00001",
  "churn_probability": 0.0,
  "churn_risk": "🟢 LOW",
  "predicted_ltv": 3340.30,
  "ltv_tier": "Mid LTV",
  "risk_score": 0.19,
  "segment": "Promising",
  "actions": [
    "Upsell to higher order value products",
    "Frequency incentives — buy 3 get 1",
    "Wishlist-based recommendations",
    "Convert to Champion tier focus"
  ]
}
```

### Example — Predict by Features
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Customer_ID": "TEST_001",
    "Recency": 5,
    "Frequency": 80,
    "Monetary": 25000.0,
    "Avg_Order_Value": 312.5,
    "Session_Count": 150,
    "Avg_Session_Duration": 35.0,
    "Pages_Viewed": 20,
    "Clicks": 60,
    "Campaign_Response": 1,
    "Wishlist_Adds": 25,
    "Cart_Abandon_Rate": 0.10,
    "Returns": 1
  }'
```

Interactive API docs available at:
`http://localhost:8000/docs`

---

## 📊 Dashboard

The Streamlit dashboard provides 6 interactive pages:

| Page | Description |
|------|-------------|
| 🏠 Overview | KPI cards, revenue distribution, LTV violin plots |
| 🔍 Customer Lookup | Live prediction, risk gauge, radar chart |
| 📊 Segment Analysis | Quadrant plot, heatmap, breakdowns |
| 💰 Revenue Recovery | ROI simulation with live sliders |
| 🤖 Model Performance | ROC curve, SHAP charts, metrics |
| 🎯 Action Center | Campaign lists, export tools |

Access at: `http://localhost:8501`

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| 01_model_comparison | Exploratory model benchmarking across 6 algorithms |
| 02_synthetic_data_experiment | Feature enrichment experiment + stress testing |
| 03_shap_analysis | Complete SHAP explainability analysis |
| 04_customer_segmentation | Segmentation development + revenue simulation |

---

## 🔍 SHAP Analysis — Key Findings

### Churn Model (LightGBM)

| Rank | Feature | SHAP Value | Insight |
|------|---------|-----------|---------|
| 1 | Avg_Order_Value | 1.0000 | Hard threshold ~$50 |
| 2 | Cart_Abandon_Rate | 0.1016 | >25% → churn signal |
| 3 | Wishlist_Adds | 0.0862 | <5 adds → churn risk |

### LTV Model (RandomForest)

| Rank | Feature | SHAP Value | Insight |
|------|---------|-----------|---------|
| 1 | Frequency | 1.0000 | >40 purchases → high LTV |
| 2 | Session_Count | 0.4275 | >75 sessions → value signal |
| 3 | Wishlist_Adds | 0.2449 | Breadth of interest |

### Cross-Model Insight

| Role | Features |
|------|---------|
| Both Models | Avg_Order_Value, Wishlist_Adds |
| Churn Only | Cart_Abandon_Rate, Avg_Session_Duration |
| LTV Only | Frequency, Session_Count, Clicks |

---

## 🎯 Customer Segmentation

### Methodology

Standard churn model threshold (0.30) was found to
create a model blind spot — high LTV customers
received near-zero churn probability due to
Avg_Order_Value dominance confirmed by SHAP analysis.

**Solution: Composite Risk Score + Within-Tier
Percentile Ranking**
```
Risk Score = 0.40 × Churn_Probability
           + 0.30 × Recency_percentile
           + 0.30 × Cart_Abandon_percentile

Segments assigned by top 25% riskiest
customers within each LTV tier
```

This approach mirrors production systems used
at companies including Amazon, Netflix and Spotify.

---

## 🐳 Docker Deployment
```bash
# Build images
docker compose build

# Start all services
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

Services:

| Service | Container | Port |
|---------|-----------|------|
| FastAPI | churn_ltv_api | 8000 |
| Streamlit | churn_ltv_app | 8501 |

---

## 🔮 Future Improvements

1. **Temporal Behaviour Modelling**
   Replace static aggregates with time-series
   features to capture engagement trends.

2. **Real-Time Churn Monitoring**
   Streaming pipeline triggering alerts when
   behavioural signals cross SHAP thresholds.

3. **Revenue Impact Simulation**
   Retention ROI simulator integrated with
   historical campaign success rates.

4. **Deep Learning Sequence Models**
   LSTM or Transformer models trained on
   transaction-level time series.

---

## 👤 Author

**Pratik Chetry**

Built as a complete end-to-end ML engineering
portfolio project demonstrating production-grade
system design, model explainability, and
business impact framing.

---

<div align="center">

⭐ If you found this project useful, please star it!

</div>
