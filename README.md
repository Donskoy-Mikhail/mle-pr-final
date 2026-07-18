# Capstone Project: E-Commerce Recommendation System

**Objective:** Build a prototype recommendation system optimized for the **add-to-cart** metric, run experiments with MLflow, deploy an inference service using FastAPI and Docker, and describe the model retraining pipeline in Airflow and the monitoring setup.

## Tech Stack

* Python 3.11
* pandas, numpy, scipy, scikit-learn
* implicit (ALS for implicit feedback)
* MLflow (experiments and artifacts)
* FastAPI + Uvicorn (inference)
* Prometheus Client (service metrics)
* Docker / Docker Compose
* Airflow

## Repository Structure

```text
ecomm_recsys_project/
├── config/
│   └── config.yaml                  # Training and service configuration
├── prometheus/
│   └── prometheus.yaml              # Prometheus configuration
├── data/                            # Directory where the data should be placed
├── ml-airflow/                      # Airflow service and model training DAG
├── src/
│   ├── common/
│   │   ├── io_utils.py
│   │   └── metrics.py
│   ├── pipeline/
│   │   ├── train.py                 # Model training and artifact saving
│   │   ├── train_cli.py             # Training CLI wrapper
│   │   └── utils.py                 # Utility functions
├── .gitignore
├── Dockerfile_ml_service            # Dockerfile for the FastAPI service
├── docker-compose.yml               # Docker Compose configuration for FastAPI, Prometheus, and Grafana
├── requirements.txt                 # Full list of dependencies
├── app.py                           # FastAPI application
├── run_mlflow_server.sh             # MLflow server startup script
├── recsys_ecommerce_capstone.ipynb  # Notebook containing experiments
├── dashboard_png.png                # Monitoring dashboard screenshot
├── dashboard.json                   # Dashboard file for importing into Grafana
└── README.md
```

# SETUP

## Data

The project expects the following three datasets to be placed in the `data` directory:

* `events.csv`
* `item_properties*.csv` (`part1`, `part2`)
* `category_tree.csv`

Make sure that an `.env` file with the following contents is located in the same directory as `docker-compose.yaml`:

```bash
DB_SOURCE_HOST=
DB_SOURCE_PORT=
DB_SOURCE_NAME=
DB_SOURCE_USER=
DB_SOURCE_PASSWORD=

DB_DESTINATION_HOST=
DB_DESTINATION_PORT=
DB_DESTINATION_NAME=
DB_DESTINATION_USER=
DB_DESTINATION_PASSWORD=

MLFLOW_S3_ENDPOINT_URL=
S3_BUCKET_NAME=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=

# No changes are required below
EXPERIMENT_NAME=course_yandex
RUN_NAME=run_yandex

APP_DOCKER_PORT=8081
APP_VM_PORT=8081
GRAFANA_PORT=3000
GRAFANA_USER="admin"
GRAFANA_PASS="grafana"
```

## Quick Start — Local Setup

```bash
cd ecomm_recsys_project
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Start MLflow
source .venv/bin/activate
export $(cat .env | xargs)
sh run_mlflow_server.sh

# Start the notebook
export $(cat .env | xargs)
jupyter notebook notebooks/recsys_ecommerce_capstone.ipynb
```

## FastAPI, Prometheus, and Grafana Services

```bash
docker compose up -d --build
```

## Airflow Service

```bash
cd ml_airflow
echo -e "\nAIRFLOW_UID=$(id -u)" >> .env
docker compose up airflow-init
docker compose down --volumes --remove-orphans
docker compose up -d --build
```

# Example FastAPI Request

```bash
curl -X POST "http://0.0.0.0:8081/recommend" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 123, "k": 20, "recent_item_ids": [111, 222, 333]}'
```

## Project Guide

### Translating the Business Objective

The objective is to maximize the probability that a user adds an item to their shopping cart.

Offline metrics:

* Recall@K
* MAP@K
* NDCG@K

Online metrics:

* CTR
* Add-to-Cart Rate
* Session-level and user-level conversion rates

### Training Infrastructure

* **MLflow:** A local server started using `run_mlflow_server.sh`. Model artifacts are stored in `/models`.
* **Configuration:** `config/config.yaml` controls file paths, model parameters, and logging settings.

### Exploratory Data Analysis

The notebook includes an analysis of:

* Event distributions
* The conversion funnel: `view → cart → purchase`
* User activity over time
* The proportions of new and returning users
* Product-level and category-level summaries

### Feature Engineering and Model Training

* Weighted interactions:

  * `view = 1`
  * `cart = 5`
  * `purchase = 10`
* Time-based data splitting: the latest interactions of each user are assigned to the validation and test sets.
* Candidate generation methods:

  * Item popularity
  * Co-visitation
* ALS model from the `implicit` library with BM25 or TF-IDF weighting.
* Metrics and model artifacts are logged to MLflow.

### Production Deployment

The FastAPI service loads the model artifacts, including:

* User factors
* Item factors
* User and item mappings

The service supports cold-start recommendations based on the `recent_item_ids` list and uses popular items as a fallback.

### Monitoring

The service exposes Prometheus metrics, including:

* Request latency
* Number of requests

### Airflow

The Airflow service allows the input datasets to be placed in the `data` directory:

* `events`
* `item_properties`
* `category_tree`

After the DAG has completed successfully, the resulting model can be found in the same directory under:

```text
data/models
```
