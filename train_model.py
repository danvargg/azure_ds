#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-09.
Author: @daniel
"""
from azureml.core import Model
import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from azureml.train.sklearn import SKLearn
from azureml.core import Experiment, Run, Model
from azureml.train.estimator import Estimator

# Get the experiment run context
run = Run.get_context()

# Using arguments
# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, dest='reg', default=0.01)
args = parser.parse_args()
reg = args.reg

# Prepare the dataset
diabetes = pd.read_csv('data.csv')
X, y = data[['Feature1', 'Feature2', 'Feature3']].values, data['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

# Train a logistic regression model
reg = 0.1
model = LogisticRegression(C=1/reg, solver="liblinear").fit(X_train, y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
run.log('Accuracy', np.float(acc))

# Save the trained model
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/model.pkl')

run.complete()

# Using an estimator
# Create an estimator
estimator = Estimator(
    source_directory='experiment_folder', entry_script='training_script.py',
    compute_target='local', conda_packages=['scikit-learn']
)

# Create and run an experiment
experiment = Experiment(workspace=ws, name='training_experiment')
run = experiment.submit(config=estimator)

# Estimator from specific framework

# Create an estimator
estimator = SKLearn(
    source_directory='experiment_folder', entry_script='training_script.py',
    script_params={'--reg_rate': 0.1}, compute_target='local'
)

# Create and run an experiment
experiment = Experiment(workspace=ws, name='training_experiment')
run = experiment.submit(config=estimator)

# list the files generated
# List the files generated by the experiment
for file in run.get_file_names():  # "run" is a reference to a completed experiment run
    print(file)

# Download a named file
run.download_file(name='outputs/model.pkl', output_file_path='model.pkl')

# Model registration enables you to track multiple versions of a model, and retrieve models for inferencing
model = Model.register(
    workspace=ws, model_name='classification_model', model_path='model.pkl',  # local path
    description='A classification model', tags={'dept': 'sales'},
    model_framework=Model.Framework.SCIKITLEARN, model_framework_version='0.20.3'
)

# From the run
run.register_model(
    model_name='classification_model', model_path='outputs/model.pkl',  # run outputs path
    description='A classification model', tags={'dept': 'sales'},
    model_framework=Model.Framework.SCIKITLEARN, model_framework_version='0.20.3'
)

# View registered models
for model in Model.list(ws):
    # Get model name and auto-generated version
    print(model.name, 'version:', model.version)