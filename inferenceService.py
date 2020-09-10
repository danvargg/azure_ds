#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-10.
Author: @daniel
"""
import json
import numpy as np
import joblib
import requests
from azureml.core import Model as model_reg
from azureml.core.model import Model, InferenceConfig
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.core.webservice import AksWebservice, LocalWebservice

# Register a trained model
classification_model = model_reg.register(
    workspace=ws, model_name='classification_model', model_path='model.pkl',  # local path
    description='A classification model'
)

# From run
run.register_model(
    model_name='classification_model', model_path='outputs/model.pkl',  # run outputs path
    description='A classification model'
)

# Init service


def init():
    # Called when the service is loaded
    global model
    # Get the path to the registered model file and load it
    model_path = Model.get_model_path('classification_model')
    model = joblib.load(model_path)


def run(raw_data):
    # Called when a request is received
    # Get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # Get a prediction from the model
    predictions = model.predict(data)
    # Return the predictions as any JSON serializable format
    return predictions.tolist()


# Create env
# Add the dependencies for your model
myenv = CondaDependencies()
myenv.add_conda_package("scikit-learn")

# Save the environment config as a .yml file
env_file = 'service_files/env.yml'
with open(env_file, "w") as f:
    f.write(myenv.serialize_to_string())

print("Saved dependency info in", env_file)

# Combining the Script and Environment in an InferenceConfig
classifier_inference_config = InferenceConfig(
    runtime="python", source_directory='service_files', entry_script="score.py", conda_file="env.yml"
)

# Define a Deployment Configuration
cluster_name = 'aks-cluster'
compute_config = AksCompute.provisioning_configuration(location='eastus')
production_cluster = ComputeTarget.create(ws, cluster_name, compute_config)
production_cluster.wait_for_completion(show_output=True)

# Deploy the config
classifier_deploy_config = AksWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=1
)

# Deploy the model
model = ws.models['classification_model']
service = Model.deploy(
    workspace=ws, name='classifier-service', models=[model],
    inference_config=classifier_inference_config, deployment_config=classifier_deploy_config,
    deployment_target=production_cluster
)
service.wait_for_deployment(show_output=True)

# Consume inference service
endpoint = service.scoring_uri
print(endpoint)

# Authentication
# Get auth keys
primary_key, secondary_key = service.get_keys()

# An array of new data cases
x_new = [[0.1, 2.3, 4.1, 2.0],
         [0.2, 1.8, 3.9, 2.1]]

# Convert the array to a serializable list in a JSON document
json_data = json.dumps({"data": x_new})

# Set the content type in the request headers
request_headers = {"Content-Type": "application/json",
                   "Authorization": "Bearer " + key_or_token}

# Call the service
response = requests.post(
    url=endpoint, data=json_data, headers=request_headers
)

# Get the predictions from the JSON response
predictions = json.loads(response.json())

# Print the predicted class for each case.
for i in range(len(x_new)):
    print(x_new[i], predictions[i])

# Check service state
# Get the deployed service
service = AciWebservice(name='classifier-service', workspace=ws)

# Check its state
print(service.state)

# Logs
print(service.get_logs())

# Deploy to a Local Container
deployment_config = LocalWebservice.deploy_configuration(port=8890)
service = Model.deploy(
    ws, 'test-svc', [model], inference_config, deployment_config
)

# Check local service
print(service.run(input_data=json_data))

# Reload it
service.reload()
print(service.run(input_data=json_data))
