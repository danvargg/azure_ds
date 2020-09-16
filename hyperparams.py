#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-15
Author: @danvargg
"""
import os
import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from azureml.core import Run, Experiment
from azureml.train.hyperdrive import (choice, normal, GridParameterSampling,
                                      RandomParameterSampling, BayesianParameterSampling,
                                      uniform, BanditPolicy, MedianStoppingPolicy,
                                      TruncationSelectionPolicy, HyperDriveConfig,
                                      PrimaryMetricGoal)

# Defining a search space
param_space = {
    '--batch_size': choice(16, 32, 64),
    '--learning_rate': normal(10, 3)
}

# Grid sampling
param_space = {
    '--batch_size': choice(16, 32, 64),
    '--learning_rate': choice(0.01, 0.1, 1.0)
}
param_sampling = GridParameterSampling(param_space)

# Random sampling
param_space = {
    '--batch_size': choice(16, 32, 64),
    '--learning_rate': normal(10, 3)
}
param_sampling = RandomParameterSampling(param_space)

# Bayesian sampling
param_space = {
    '--batch_size': choice(16, 32, 64),
    '--learning_rate': uniform(0.5, 0.1)
}
param_sampling = BayesianParameterSampling(param_space)

# Configuring early termination
# Bandit policy
early_termination_policy = BanditPolicy(
    slack_amount=0.2, evaluation_interval=1, delay_evaluation=5
)

# Median stopping policy
early_termination_policy = MedianStoppingPolicy(
    evaluation_interval=1, delay_evaluation=5
)

# Truncation selection policy
early_termination_policy = TruncationSelectionPolicy(
    truncation_percentage=10, evaluation_interval=1, delay_evaluation=5
)

# Creating a training script for hyperparameter tuning
# Get regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--regularization', type=float,
                    dest='reg_rate', default=0.01)
args = parser.parse_args()
reg = args.reg_rate

# Get the experiment run context
run = Run.get_context()

# load the training dataset
data = run.input_datasets['training_data'].to_pandas_dataframe()

# Separate features and labels, and split for training/validatiom
X = data[['feature1', 'feature2', 'feature3', 'feature4']].values
y = data['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a logistic regression model with the reg hyperparameter
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate and log accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()

# Configuring and running a hyperdrive experiment
# Assumes ws, sklearn_estimator and param_sampling are already defined
hyperdrive = HyperDriveConfig(
    estimator=sklearn_estimator, hyperparameter_sampling=param_sampling, policy=None,
    primary_metric_name='Accuracy', primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
    max_total_runs=6, max_concurrent_runs=4
)
experiment = Experiment(workspace=ws, name='hyperdrive_training')
hyperdrive_run = experiment.submit(config=hyperdrive)

# Monitoring and reviewing hyperdrive runs
# Retrieve the logged metrics
for child_run in run.get_children():
    print(child_run.id, child_run.get_metrics())

# List all runs
for child_run in hyperdrive_.get_children_sorted_by_primary_metric():
    print(child_run)

# Retrieve the best performing run
best_run = hyperdrive_run.get_best_run_by_primary_metric()
