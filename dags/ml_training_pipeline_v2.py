from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

# Add src to path so DAGs can import ml_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.breast_cancer import train_model, eval_model, promote_model

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="ml_training_pipeline_v2",
    default_args=default_args,
    description="Pipeline: train,evaluate and promote model",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    eval_task = PythonOperator(
        task_id="eval_model",
        python_callable=eval_model,
    )


    promote_task = PythonOperator(
        task_id="promote_model",
        python_callable=promote_model,
    )

    train_task >> eval_task >> promote_task