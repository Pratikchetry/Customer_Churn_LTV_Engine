
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

# 📋 Overview

The **Customer Churn & LTV Prediction Engine** is a production-grade machine learning system that predicts **customer churn risk** and **customer lifetime value (LTV)** to help businesses prioritize retention strategies and maximize revenue.

Instead of looking at churn and value independently, this system combines both models to produce **actionable customer intelligence**.

The system enables businesses to:

• Identify customers likely to churn  
• Estimate the financial impact of churn  
• Segment customers by value and behavioral risk  
• Prioritize retention campaigns based on ROI  
• Serve predictions through an API  
• Explore insights through an interactive dashboard  

This project demonstrates the **full ML engineering lifecycle**:

Raw data → feature engineering → model training → explainability → segmentation → API deployment → dashboard → containerization.

---

# 🏆 Verified Results

## Model Performance

| Model | Algorithm | Metric | Score |
|------|-----------|-------|------|
| Churn | LightGBM | F1 Score | **0.8770** |
| Churn | LightGBM | ROC-AUC | **0.9836** |
| Churn | LightGBM | Recall | **1.0000** |
| Churn | LightGBM | Precision | **0.7810** |
| LTV | RandomForest | R² | **0.9226** |
| LTV | RandomForest | RMSE | **$2,806** |

These results indicate strong predictive performance, particularly for the churn model which prioritizes **recall to ensure high-risk customers are not missed**.

---

# 👥 Customer Segmentation

The system segments **10,000 customers** into six behavioral groups based on predicted churn risk and lifetime value.

| Segment | Customers | Avg LTV | Total LTV |
|-------|--------|--------|--------|
| 🟢 Champion | 2,472 | $21,231 | $52.5M |
| 🟡 At-Risk VIP | 828 | $11,106 | $9.2M |
| 🔵 Promising | 2,550 | $3,725 | $9.5M |
| 🟠 Vulnerable | 850 | $2,093 | $1.8M |
| ⚪ Hibernating | 2,474 | $547 | $1.4M |
| 🔴 Losing Customer | 826 | $75 | $0.1M |

Total revenue modeled: **$74,372,087**

Revenue currently at risk: **$11,037,006 (14.8%)**

---

# 💰 Revenue Recovery Simulation

Retention campaign simulations estimate potential revenue recovery.

| Segment | Investment | Net ROI | ROI % |
|-------|-----------|-------|-------|
| At-Risk VIP | $41,400 | $4.10M | **9,896%** |
| Vulnerable | $21,250 | $0.60M | **2,830%** |
| Losing Customer | $4,130 | $0.01M | **126%** |
| **Total** | **$66,780** | **$4.70M** | **7,043%** |

This simulation highlights that **small targeted retention investments can protect millions in revenue**.

---

# 🏗 System Architecture

```
Raw Data
   ↓
Feature Engineering
   ↓
┌───────────────────┬──────────────────┐
│  Churn Model      │   LTV Model      │
│  LightGBM         │   RandomForest   │
│  F1 = 0.8770      │   R² = 0.9226    │
│  AUC = 0.9836     │   RMSE = $2,806  │
└───────────────────┴──────────────────┘
   ↓
SHAP Explainability
   ↓
Customer Segmentation
(Composite Risk Score +
Within-Tier Percentile Ranking)
   ↓
FastAPI REST API
   ↓
Streamlit Dashboard
   ↓
Docker Compose Deployment
```

Total pipeline runtime: **19.1 seconds end-to-end**

---

# 🛠 Technology Stack

| Layer | Technology |
|------|------------|
| Programming Language | Python 3.12 |
| Churn Model | LightGBM |
| LTV Model | RandomForest (Scikit-learn) |
| Explainability | SHAP |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker + Docker Compose |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly |

---

# 📁 Project Structure

```
Customer_Churn_LTV_Engine/

data/
│
models/
│
notebooks/
│   ├── model_experiments.ipynb
│   ├── shap_analysis.ipynb
│   └── segmentation_analysis.ipynb
│
src/
│   ├── feature_engineering.py
│   ├── train_churn_model.py
│   ├── train_ltv_model.py
│   ├── customer_segmentation.py
│   ├── predict.py
│   └── pipeline.py
│
api/
│   └── main.py
│
app/
│   └── streamlit_app.py
│
reports/
│   └── shap_visualizations
│
docker/
│   ├── Dockerfile.api
│   └── Dockerfile.app
│
docker-compose.yml
requirements.txt
README.md
```

---

# ⚡ Quick Start

Run the complete pipeline locally in **five commands**.

```bash
git clone https://github.com/YOUR_USERNAME/Customer_Churn_LTV_Engine.git
cd Customer_Churn_LTV_Engine
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/pipeline.py
```

---

# 🌐 REST API

The system exposes predictions through **FastAPI**.

| Method | Endpoint | Description |
|------|---------|-------------|
| GET | /health | API health check |
| GET | /predict/{customer_id} | Predict churn and LTV by ID |
| POST | /predict | Predict using feature JSON |
| GET | /segment/summary | Segment summary |

Example request:

```bash
curl http://localhost:8000/predict/CUST00001
```

Example response:

```json
{
  "customer_id": "CUST00001",
  "churn_probability": 0.0,
  "churn_risk": "LOW",
  "predicted_ltv": 3340.30,
  "ltv_tier": "Mid LTV",
  "risk_score": 0.19,
  "segment": "Promising"
}
```

---

# 📊 Dashboard

The **Streamlit dashboard** provides interactive analysis tools.

Key sections include:

• Customer prediction lookup  
• Segment distribution analysis  
• Revenue recovery simulation  
• Model performance metrics  
• SHAP explainability plots  

Dashboard URL:

```
http://localhost:8501
```

---

# 🔍 Key Engineering Decisions

### 1. Churn Threshold Selection

A threshold of **0.30** was chosen using **F-beta optimization** to prioritize recall.

In churn prediction, **missing a churner is more costly than incorrectly flagging a loyal customer**.

---

### 2. Composite Risk Score

```
Risk Score =
0.40 × Churn Probability
0.30 × Recency Percentile
0.30 × Cart Abandonment Percentile
```

This approach addresses a **model blind spot discovered through SHAP analysis** where high-value customers had artificially low churn probability.

---

### 3. Within-Tier Segmentation

Customers are ranked by risk **within their LTV tier** rather than using a global threshold.

This ensures high-value segments like **At-Risk VIP** are always identified.

This strategy mirrors approaches used by companies such as **Amazon, Netflix, and Spotify**.

---

### 4. Data Leakage Prevention

The following features were removed after correlation analysis revealed leakage:

• Recency  
• Monetary  
• LTV  
• Engagement Score  

Removing these features ensured **realistic model performance**.

---

# 🔎 SHAP Explainability Insights

### Churn Model

| Rank | Feature | Insight |
|----|----|----|
| 1 | Avg Order Value | strong churn threshold |
| 2 | Cart Abandon Rate | high abandonment signals churn |
| 3 | Wishlist Adds | low wishlist activity increases churn risk |

### LTV Model

| Rank | Feature | Insight |
|----|----|----|
| 1 | Frequency | strong indicator of customer value |
| 2 | Session Count | engagement proxy |
| 3 | Wishlist Adds | breadth of product interest |

These insights informed the **composite risk scoring approach used for segmentation**.

---

# 🔮 Future Improvements

• Time-series behavioral modeling  
• Real-time churn monitoring pipelines  
• Campaign ROI simulation with historical data  
• Deep learning sequence models (LSTM / Transformers)

---

# 👤 Author

**Pratik Chetry**
**Email ID:chetrypratik2002@gmail.com**

This project demonstrates practical experience in:

• Machine learning system design  
• explainable AI  
• customer segmentation  
• business impact modeling  
• API and dashboard deployment  

---

<div align="center">

⭐ If you found this project useful, please consider starring the repository.

</div>
