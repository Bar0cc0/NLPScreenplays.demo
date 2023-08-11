#!/usr/bin/env python3

import pathlib
from datetime import timedelta
import pandas as pd
from airflow.models.dag import DAG
from airflow.utils.task_group import TaskGroup
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator 
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.task_group import TaskGroup



# postgresql conn id
pg_conn_args = {
    "owner": "Bar0cc0",
    "start_date": days_ago(0),
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "template_searchpath":"~/airflow",
    "wait_for_downstream": True,
    "catchup": False,
    "conn_id": "postgres_default",
    "conn_type": "postgres",
    "host": "localhost",
    "login": "postgres",
    "password": "secret123",
    "port": 5432,
}

# $AIRFLOW_HOME=~/airflow
# which must contains both ./dags and ./marto 
PATH = pathlib.Path(__file__).parents[1].joinpath('marto')


def _insert_table():
    # reads results 
    table = pd.read_excel(f"{PATH}/data/bttf.xlsx")
    # write to postgres db
    for row in table.itertuples():
        SQLExecuteQueryOperator(
            task_id = f"insert_row_{row[0]}",
            hook_params={"schema":"postgres"},
            sql = f"""
                INSERT INTO bttf 
                VALUES (
                    {row[1]},'{row[2]}','{row[3]}','{row[4]}'
                );"""
        )


# dag 
with DAG(dag_id="ETL_pipeline_adhoc", default_args = pg_conn_args):
    
    # Create mock dataset
    with TaskGroup("extract") as extract:
        dataset = BashOperator(
            task_id="dataset_gen",
            bash_command=f"python3 {PATH}/CreateDataset.py"        
        )
        log = EmptyOperator(task_id="log_dataset_gen")
        dataset >> log
    
    # Analyse data
    with TaskGroup("transform") as transform:
        topic_classification = BashOperator(
            task_id="topic_classification",
            bash_command=f"python3 {PATH}/MarTo.py"
        )
        log = EmptyOperator(task_id="log_topic_class")
        topic_classification >> log

    # Loading data into db table is conditioned by previous steps
    # => must wait for the .xlsx file to be available
    wait_file = FileSensor(task_id="wait_for_file", 
                           filepath=PATH.joinpath("data/bttf.xlsx"), 
                           )
    
    # Write db
    with TaskGroup("load") as load:
        create_table = SQLExecuteQueryOperator(
            task_id = "create_table",
            hook_params={"schema":"postgres"},
            sql = """
                DROP TABLE IF EXISTS bttf;
                CREATE TABLE bttf(
                    index INTEGER PRIMARY KEY,
                    timecode TIME,
                    part TEXT,
                    dialogue TEXT
                );"""
        )

        insert_table = PythonOperator(
            task_id="insert_table",
            python_callable=_insert_table
        )   
        
        log = EmptyOperator(task_id="log_db")
        create_table >> insert_table >> log

    # Sequential pipeline 
    extract >> transform >> wait_file >> load

