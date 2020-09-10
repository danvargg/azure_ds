#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-10.
Author: @daniel
"""
from azureml.pipeline.core import Schedule
from azureml.core import Datastore
import requests
from azureml.core import Experiment
from azureml.pipeline.core import Pipeline, ScheduleRecurrence, Schedule, PipelineData
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep

# Define pipeline steps
# Step to run a Python script
step1 = PythonScriptStep(
    name='prepare data', source_directory='scripts', script_name='data_prep.py',
    compute_target='aml-cluster', runconfig=run_config
)

# Step to run an estimator
step2 = EstimatorStep(
    name='train model', estimator=sk_estimator, compute_target='aml-cluster'
)

# Construct the pipeline
train_pipeline = Pipeline(workspace=ws, steps=[step1, step2])

# Create an experiment and run the pipeline
experiment = Experiment(workspace=ws, name='training-pipeline')
pipeline_run = experiment.submit(train_pipeline)

# Data Pipeline
# Get a dataset for the initial data
raw_ds = Dataset.get_by_name(ws, 'raw_dataset')

# Define a PipelineData object to pass data between steps
data_store = ws.get_default_datastore()
prepped_data = PipelineData('prepped',  datastore=data_store)

# Step to run a Python script
step1 = PythonScriptStep(
    name='prepare data', source_directory='scripts', script_name='data_prep.py',
    compute_target='aml-cluster', runconfig=run_config,
    # Specify dataset as initial input
    inputs=[raw_ds.as_named_input('raw_data')],
    # Specify PipelineData as output
    outputs=[prepped_data],
    # Also pass as data reference to script
    arguments=['--folder', prepped_data]
)

# Step to run an estimator
step2 = EstimatorStep(
    name='train model', estimator=sk_estimator, compute_target='aml-cluster',
    # Specify PipelineData as input
    inputs=[prepped_data],
    # Pass as data reference to estimator script
    estimator_entry_script_arguments=['--folder', prepped_data]
)

# Reuse pipeline steps
step1 = PythonScriptStep(
    name='prepare data', source_directory='scripts', script_name='data_prep.py',
    compute_target='aml-cluster', runconfig=run_config,
    inputs=[raw_ds.as_named_input('raw_data')], outputs=[prepped_data],
    arguments=['--folder', prepped_data],
    allow_reuse=False  # Disable step reuse
)

# Publish pipeline
published_pipeline = pipeline.publish(
    name='training_pipeline', description='Model training pipeline', version='1.0'
)

# Get the most recent run of the pipeline
pipeline_experiment = ws.experiments.get('training-pipeline')
run = list(pipeline_experiment.get_runs())[0]

# Publish the pipeline from the run (after successfull)
published_pipeline = run.publish_pipeline(
    name='training_pipeline', description='Model training pipeline', version='1.0'
)

# URI of published pipeline
rest_endpoint = published_pipeline.endpoint
print(rest_endpoint)

# Use published pipeline
response = requests.post(
    rest_endpoint, headers=auth_header,
    json={
        "ExperimentName": "run_training_pipeline"
    }
)
run_id = response.json()["Id"]
print(run_id)

# Define params for a pipeline
reg_param = PipelineParameter(name='reg_rate', default_value=0.01)

step2 = EstimatorStep(
    name='train model', estimator=sk_estimator, compute_target='aml-cluster',
    inputs=[prepped],
    estimator_entry_script_arguments=['--folder', prepped, '--reg', reg_param]
)

# Run pipeline with parameter
response = requests.post(
    rest_endpoint, headers=auth_header,
    json={
        "ExperimentName": "run_training_pipeline", "ParameterAssignments": {"reg_rate": 0.1}
    }
)

# Schedule pipelines
daily = ScheduleRecurrence(frequency='Day', interval=1)
pipeline_schedule = Schedule.create(
    ws, name='Daily Training', description='trains model every day',
    pipeline_id=published_pipeline.id, experiment_name='Training_Pipeline', recurrence=daily
)

# On data changes

training_datastore = Datastore(workspace=ws, name='blob_data')
pipeline_schedule = Schedule.create(
    ws, name='Reactive Training', description='trains model on data change',
    pipeline_id=published_pipeline_id, experiment_name='Training_Pipeline',
    datastore=training_datastore, path_on_datastore='data/training'
)
