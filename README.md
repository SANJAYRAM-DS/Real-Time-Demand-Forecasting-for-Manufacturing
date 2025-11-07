# Real-Time Demand Forecasting for Manufacturing (M5 Forecasting Project)

> **End-to-End Production-Level EDA → Forecasting → MLOps System**  
> Built using Python, Pandas, NumPy, Matplotlib, LightGBM, and Airflow — designed to predict daily demand across thousands of SKUs and stores, detect bottlenecks, and provide explainable insights for production planning.

---

## Project Overview

This project implements a **real-time demand forecasting system** for the manufacturing / retail sector using the **M5 Forecasting dataset (Walmart)** as a base.

It covers the full lifecycle of a data science solution:

1. **Data Ingestion & Integration**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering**
4. **Baseline & ML Model Building**
5. **Visualization & Reporting (Dashboards)**
6. **Monitoring & Drift Detection**
7. **MLOps Deployment (Airflow + Docker + MLflow)**
8. **Iterative Improvement & Error Forensics**

The workflow mirrors how an industrial AI team would deliver a forecasting solution for real production use.

---

## Business Problem

Manufacturers and retailers often face:
- Over-production → excess inventory and storage cost  
- Under-production → stockouts and lost revenue  
- Seasonal, price, and event-driven demand shifts  

**Goal:** Build an automated system that forecasts **daily demand 28 days ahead** per product × store, while providing actionable insights (bottlenecks, bias, event impact, and forecast reliability).

---

## Dataset — [M5 Forecasting Accuracy (Kaggle)](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data)

**Files Used**
| File | Description |
|------|--------------|
| `sales_train_validation.csv` | Daily unit sales per item/store (1913 days) |
| `sales_train_evaluation.csv` | Extended version for evaluation (1941 days) |
| `calendar.csv` | Date mapping + events + SNAP days |
| `sell_prices.csv` | Weekly prices for each item/store |
| `sample_submission.csv` | Submission format (forecast horizon 28 days) |

---

## Architecture & Workflow

```text
Raw Data (CSV)
     │
     ▼
[1] Data Ingestion & Integration  → merge sales + calendar + prices
     │
     ▼
[2] Exploratory Data Analysis (EDA)
     │
     ▼
[3] Feature Engineering  → lag, rolling, event, price features
     │
     ▼
[4] Model Training (LightGBM)
     │
     ▼
[5] Forecast Generation (28 days horizon)
     │
     ▼
[6] Visualization (Matplotlib / Streamlit)
     │
     ▼
[7] Monitoring (Evidently + MLflow + Airflow)
     │
     ▼
[8] Continuous Improvement (Champion–Challenger, Drift, Bias)
````

---

## Key Features

-Multi-table integration (sales + prices + events)
- Lag, rolling, and time-based feature engineering
- LightGBM global forecasting model
- Baselines: Naïve, Moving Average, Exponential Smoothing
- Daily monitoring (MAE, RMSE, MAPE, bias, drift)
- Streamlit interactive dashboard
- Automated pipeline (Airflow / cron + Docker)
- MLflow model registry & versioning
- Data validation (Pandera + Great Expectations)

---

## Tech Stack

| Layer            | Tools & Libraries                  |
| ---------------- | ---------------------------------- |
| **Language**     | Python 3.11                        |
| **EDA**          | Pandas, NumPy, Matplotlib, Seaborn |
| **Modeling**     | LightGBM, XGBoost, Statsmodels     |
| **Validation**   | Pandera, Great Expectations        |
| **Monitoring**   | Evidently, MLflow                  |
| **Scheduling**   | Apache Airflow / cron              |
| **Dashboarding** | Streamlit                          |
| **Environment**  | Docker, Makefile                   |
| **CI/CD**        | GitHub Actions, pytest             |

---

## Folder Structure

```bash
m5_forecasting_project/
├── data/
│   ├── raw/                # Original M5 CSVs
│   ├── interim/            # Cleaned & merged
│   ├── processed/          # Features, train/valid sets
│   └── external/           # Extra sources (optional)
│
├── notebooks/              # Development notebooks (Steps 1–8)
│   ├── 01_data_collection_integration.ipynb
│   ├── 02_eda_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_building.ipynb
│   ├── 05_reporting_visualization.ipynb
│   ├── 06_monitoring_mlop.ipynb
│   └── 07_iterative_improvement.ipynb
│
├── pipelines/              # Production code (used by Airflow)
│   ├── 01_ingest.py
│   ├── 02_validate.py
│   ├── 03_features.py
│   ├── 04_predict.py
│   ├── 05_evaluate.py
│   ├── 06_publish.py
│   └── utils_*.py
│
├── models/
│   ├── lightgbm/model.txt
│   ├── baselines/
│   └── registry/model_registry.csv
│
├── reports/
│   ├── eda_summary.pdf
│   ├── metrics_summary.csv
│   ├── drift_latest.html
│   ├── feature_doc.md
│   └── model_card.md
│
├── dashboards/
│   ├── app.py              # Streamlit app
│   ├── components/
│   └── assets/
│
├── monitoring/
│   ├── drift_reports/
│   ├── metrics/
│   └── alerts/
│
├── tests/
│   ├── test_data_validation.py
│   ├── test_feature_engineering.py
│   └── test_pipeline_e2e.py
│
├── configs/
│   ├── model_config.yaml
│   ├── data_schema.yaml
│   ├── drift_config.yaml
│   └── alerting_config.yaml
│
├── airflow/dags/m5_forecast_dag.py
├── logs/
├── README.md
├── requirements.txt
├── Dockerfile
└── Makefile
```

---

## Setup & Installation

### 1️.Clone the repository

```bash
git clone https://github.com/<your-username>/m5_forecasting_project.git
cd m5_forecasting_project
```

### 2️.Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # (Windows: .venv\Scripts\activate)
```

### 3️.Install dependencies

```bash
pip install -r requirements.txt
```

### 4.(Optional) Run with Docker

```bash
docker build -t m5-forecast:latest .
docker run -it --rm m5-forecast:latest
```

---

## Running the Project

### A. Run full pipeline locally

```bash
make run DATE=2025-11-07
```

or directly:

```bash
python pipelines/01_ingest.py
python pipelines/02_validate.py
python pipelines/03_features.py
python pipelines/04_predict.py
python pipelines/05_evaluate.py
python pipelines/06_publish.py
```

###  B. Launch the dashboard

```bash
streamlit run dashboards/app.py
```

Open in browser: [http://localhost:8501](http://localhost:8501)

### C. Run tests

```bash
pytest -q
```

---

## Key Metrics (sample from validation)

| Metric         | Value |
| -------------- | ----- |
| MAE            | 1.82  |
| RMSE           | 2.14  |
| MAPE           | 8.9%  |
| WRMSSE         | 0.91  |
| Coverage (80%) | 78.5% |

---

## Monitoring & Drift

* **Evidently Reports:** generated daily in `/monitoring/drift_reports/`
* **Metrics JSON:** saved under `/monitoring/metrics/`
* **Slack/Email Alerts:** configured via `configs/alerting_config.yaml`
* **MLflow Tracking:** logs all experiments and models (`mlruns/` or remote server)

---

## Continuous Improvement Loop

1. Review daily accuracy reports
2. Identify top error segments (store/category)
3. Add or adjust features (promo, event, lag)
4. Retrain challenger model
5. Shadow test 7 days → promote if improved
6. Update `model_card.md` + registry

---

## Deployment Options

| Environment         | Tool                              | Notes                              |
| ------------------- | --------------------------------- | ---------------------------------- |
| Batch (recommended) | Airflow / Cron                    | Run pipeline daily                 |
| Web                 | Streamlit                         | Interactive analytics              |
| API (optional)      | FastAPI                           | REST endpoint for forecast queries |
| Cloud               | Docker on AWS ECS / GCP Cloud Run | Containerized scalable deployment  |

---

## Authors

**SanjayRam M.**
Data Science & AI Engineering Enthusiast
Built under mentorship of **ChatGPT (GPT-5)** — full E2E walkthrough.

---

## License

MIT License © 2025 SanjayRam Manivel.
You may freely use, modify, and distribute this project with attribution.

---

## Summary

- Complete real-world manufacturing forecasting pipeline
- From raw CSVs → EDA → features → model → dashboard → MLOps
- Production-ready folder structure + reproducible Docker/CI setup
- Extendable for retail, logistics, or energy forecasting use cases

---
