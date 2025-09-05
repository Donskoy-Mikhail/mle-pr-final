from airflow.decorators import dag, task
from airflow import DAG
from airflow.operators.python import PythonOperator
from pipeline.train import train_pipeline
from pipeline.utils import load_raw_data, basic_filters, downsample_users, temporal_split_per_user
from pipeline.utils import build_mappings, build_interaction_matrix, build_user_targets, popularity_top_items
import pendulum
import os, json, yaml, numpy as np, pandas as pd
from pathlib import Path

def train_model():
    data_dir = Path(os.environ.get("DATA_DIR", "/opt/airflow/data"))
    config_dir = Path(os.environ.get("CONFIG_DIR", "/opt/airflow/config"))
    with open(config_dir / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    events, props, cats = load_raw_data(data_dir)
    result = train_pipeline(cfg, events, props, cats, data_dir / 'models')
    print(result['metrics'])

with DAG(
    dag_id='train_rec_model',
    schedule='@once',
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    tags=["ETL"]) as dag:
    
    train_model_step = PythonOperator(task_id='train_model', python_callable=train_model)

    train_model_step