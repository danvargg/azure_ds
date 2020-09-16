#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-15
Author: @daniel
"""
import os
import joblib
import requests

import numpy as np
import pandas as pd
from azureml.pipeline.core import ScheduleRecurrence, Schedule
from azureml.core import Model, Pipeline, PipelineData, Experiment
from azureml.pipeline.steps import ParallelRunConfig, ParallelRunStep

# Register / reference a model from run
classification_model = Model.register(
    workspace=ws, model_name='classification_model', model_path='model.pkl',  # local path
    description='A classification model'
)

run.register_model(
    model_name='classification_model', model_path='outputs/model.pkl',  # run outputs path
    description='A classification model'
)

# Create a scoring script


def init():
    # Use the `init` function to load the model from the model registry,
    # Runs when the pipeline step is initialized
    global model

    # load the model
    model_path = Model.get_model_path('classification_model')
    model = joblib.load(model_path)


def run(mini_batch):
    # Use the `run` function to generate predictions from each batch of data and return the results
    # This runs for each batch
    resultList = []

    # process each file in the batch
    for f in mini_batch:
        # Read comma-delimited data into an array
        data = np.genfromtxt(f, delimiter=',')
        # Reshape into a 2-dimensional array for model input
        prediction = model.predict(data.reshape(1, -1))
        # Append prediction to results
        resultList.append("{}: {}".format(os.path.basename(f), prediction[0]))
    return resultList


# Create a pipeline with a ParallelRunStep
# Get the batch dataset for input
batch_data_set = ws.datasets('batch-data')

# Set the output location
default_ds = ws.get_default_datastore()
output_dir = PipelineData(name='inferences',
                          datastore=default_ds,
                          output_path_on_compute='results')

# Define the parallel run step step configuration
parallel_run_config = ParallelRunConfig(
    source_directory='batch_scripts', entry_script="batch_scoring_script.py",
    mini_batch_size="5", error_threshold=10, output_action="append_row",
    environment=batch_env, compute_target=aml_cluster, node_count=4
)

# Create the parallel run step
parallelrun_step = ParallelRunStep(
    name='batch-score', parallel_run_config=parallel_run_config,
    inputs=[batch_data_set.as_named_input('batch_data')], output=output_dir, arguments=[],
    allow_reuse=True
)
# Create the pipeline
pipeline = Pipeline(workspace=ws, steps=[parallelrun_step])

# Run the pipeline and retrieve the step output
# Run the pipeline as an experiment
pipeline_run = Experiment(ws, 'batch_prediction_pipeline').submit(pipeline)
pipeline_run.wait_for_completion(show_output=True)

# Get the outputs from the first (and only) step
prediction_run = next(pipeline_run.get_children())
prediction_output = prediction_run.get_output_data('inferences')
prediction_output.download(local_path='results')

# Find the parallel_run_step.txt file
for root, dirs, files in os.walk('results'):
    for file in files:
        if file.endswith('parallel_run_step.txt'):
            result_file = os.path.join(root, file)

# Load and display the results
df = pd.read_csv(result_file, delimiter=":", header=None)
df.columns = ["File", "Prediction"]
print(df)

# Publishing a batch inference pipeline
published_pipeline = pipeline_run.publish_pipeline(
    name='Batch_Prediction_Pipeline', description='Batch pipeline', version='1.0'
)
rest_endpoint = published_pipeline.endpoint

# Initiate a batch inferencing job
response = requests.post(
    rest_endpoint, headers=auth_header, json={
        "ExperimentName": "Batch_Prediction"}
)
run_id = response.json()["Id"]

# Schedule the published pipeline to have it run automatically
weekly = ScheduleRecurrence(frequency='Week', interval=1)
pipeline_schedule = Schedule.create(
    ws, name='Weekly Predictions', description='batch inferencing',
    pipeline_id=published_pipeline.id, experiment_name='Batch_Prediction', recurrence=weekly
)
