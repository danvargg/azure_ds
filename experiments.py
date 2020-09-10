#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-07.
Author: @daniel
"""
import json
from azureml.core import Experiment, RunConfiguration, ScriptRunConfig
from azureml.widgets import RunDetails

# When submitting an experiment, use its run context to initialize and end
# the experiment run that is tracked in Azure Machine Learning

# create an experiment variable
experiment = Experiment(workspace=ws, name='my-experiment')

# start the experiment
run = experiment.start_logging()

# load the dataset and count the rows
data = pd.read_csv('data.csv')
row_count = (len(data))

# Save a sample of the data
os.makedirs('outputs', exist_ok=True)
data.sample(100).to_csv('outputs/sample.csv', index=False, header=True)

# Log the row count
run.log('observations', row_count)

# experiment code goes here

# end the experiment
run.complete()

# Log types
# log: Record a single named value.
# log_list: Record a named list of values.
# log_row: Record a row with multiple columns.
# log_table: Record a dictionary as a table.
# log_image: Record an image file or a plot.

# View the metrics logged by an experiment run
RunDetails(run).show()

# Retrieve the logged metrics
metrics = run.get_metrics()
print(json.dumps(metrics, indent=4))

# Upload local files to the run's outputs folder
run.upload_file(name='outputs/sample.csv', path_or_stream='./sample.csv')

# Retrieve a list of output files
files = run.get_file_names()
print(json.dumps(files, indent=4))

# Run an experiment based on a script in the experiment_files folder
# create a new RunConfig object
experiment_run_config = RunConfiguration()
experiment_folder = './experiment'

# Create a script config
script_config = ScriptRunConfig(
    source_directory=experiment_folder, script='experiment.py', run_config=experiment_run_config
)

# submit the experiment
experiment = Experiment(workspace=ws, name='my-experiment')
run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)
