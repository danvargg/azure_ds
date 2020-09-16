#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-15
Author: @danvargg
"""

import datetime as dt

from azureml.widgets import RunDetails
from azureml.core import Model, Dataset, Experiment, Run
from azureml.datadrift import DataDriftDetector, AlertConfiguration
from azureml.monitoring import ModelDataCollector
from azureml.core.webservice import AksWebservice

# Monitor data drift by comparing datasets
monitor = DataDriftDetector.create_from_datasets(
    workspace=ws, name='dataset-drift-detector', baseline_data_set=train_ds,
    target_data_set=new_data_ds, compute_target='aml-cluster', frequency='Week',
    feature_list=['age', 'height', 'bmi'], latency=24)

backfill = monitor.backfill(
    dt.datetime.now() - dt.timedelta(weeks=6), dt.datetime.now())

# Monitor data drift in service inference data
# Register the baseline dataset with the model
model = Model.register(
    workspace=ws, model_path='./model/model.pkl', model_name='my_model',
    datasets=[(Dataset.Scenario.TRAINING, train_ds)])

# Enable data collection for the deployed model


def init():
    global model, data_collect, predict_collect
    model_name = 'my_model'
    model = joblib.load(Model.get_model_path(model_name))
    # Enable collection of data and predictions
    data_collect = ModelDataCollector(
        model_name, designation='inputs', features=['age', 'height', 'bmi'])
    predict_collect = ModelDataCollector(
        model_name, designation='predictions', features=['prediction'])


def run(raw_data):
    data = json.loads(raw_data)['data']
    predictions = model.predict(data)
    # collect data and predictions
    data_collect(data)
    predict_collect(predictions)

    return predictions.tolist()


# Enable data collection in the deployment configuration
dep_config = AksWebservice.deploy_configuration(collect_model_data=True)

# Configure data drift detection
# create a new DataDriftDetector object for the deployed model
model = ws.models['my_model']
datadrift = DataDriftDetector.create_from_model(
    ws, model.name, model.version, services=['my-svc'], frequency="Week")

# or specify existing compute cluster
run = datadrift.run(
    target_date=dt.today(), services=['my-svc'],
    feature_list=['age', 'height', 'bmi'], compute_target='aml-cluster')

# show details of the data drift run
exp = Experiment(ws, datadrift._id)
dd_run = Run(experiment=exp, run_id=run.id)
RunDetails(dd_run).show()

# Configure alerts
alert_email = AlertConfiguration('data_scientists@contoso.com')
monitor = DataDriftDetector.create_from_datasets(
    ws, 'dataset-drift-detector', baseline_data_set, target_data_set,
    compute_target=cpu_cluster, frequency='Week', latency=2, drift_threshold=.3,
    alert_configuration=alert_email)
