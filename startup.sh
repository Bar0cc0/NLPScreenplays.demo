#!/bin/bash

echo -e "AIRFLOW_UID=$(id -u)" > .env
export AIRFLOW_HOME=/home/airflow
mkdir -p $AIRFLOW_HOME/dags $AIRFLOW_HOME/logs $AIRFLOW_HOME/config $AIRFLOW_HOME/marto
cp ./Rflow.py $AIRFLOW_HOME/dags
cp -R ./ $AIRFLOW_HOME/marto
sudo -u root service postgresql start
airflow standalone
