#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-15
Author: @danvargg
"""
from azureml.core.experiment import Experiment
from azureml.train.automl import AutoMLConfig
from azureml.train.automl.utilities import get_primary_metrics

# Configuring an Automated Machine Learning Experiment
automl_run_config = RunConfiguration(framework='python')
automl_config = AutoMLConfig(
    name='Automated ML Experiment', task='classification', primary_metric='AUC_weighted',
    compute_target=aml_compute, training_data=train_dataset, validation_data=test_dataset,
    label_column_name='Label', featurization='auto', iterations=12, max_concurrent_iterations=4
)

# Specifying the Primary Metric
get_primary_metrics('classification')

# Submitting an Automated Machine Learning Experiment

automl_experiment = Experiment(ws, 'automl_experiment')
automl_run = automl_experiment.submit(automl_config)

# Retrieving the Best Run and its Model
best_run, fitted_model = automl_run.get_output()
best_run_metrics = best_run.get_metrics()
for metric_name in best_run_metrics:
    metric = best_run_metrics[metric_name]
    print(metric_name, metric)

# Exploring Preprocessing Steps
for step_ in fitted_model.named_steps:
    print(step)
