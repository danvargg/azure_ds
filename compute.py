#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-10.
Author: @daniel
"""
from azureml.core.compute_target import ComputeTargetException
from azureml.core.compute import ComputeTarget, AmlCompute, DatabricksCompute
from azureml.core import Workspace, Environment
from azureml.core.conda_dependencies import CondaDependencies
from azureml.train.estimator import Estimator

# Create Azure ML env
env = Environment.from_conda_specification(
    name='training_environment', file_path='./conda.yml'
)

# Create env with specific packages
env = Environment('training_environment')
deps = CondaDependencies.create(
    conda_packages=['scikit-learn', 'pandas', 'numpy'], pip_packages=['azureml-defaults']
)
env.python.conda_dependencies = deps

# Register env
env.register(workspace=ws)

# View envs
env_names = Environment.list(workspace=ws)
for env_name in env_names:
    print('Name:', env_name)

# Retrieve and use env
training_env = Environment.get(workspace=ws, name='training_environment')
estimator = Estimator(
    source_directory='experiment_folder' entry_script='training_script.py',
    compute_target='local', environment_definition=training_env
)

# Create a managed compute target
# Load the workspace from the saved config file
ws = Workspace.from_config()

# Specify a name for the compute (unique within the workspace)
compute_name = 'aml-cluster'

# Define compute configuration
compute_config = AmlCompute.provisioning_configuration(
    vm_size='STANDARD_DS12_V2', min_nodes=0, max_nodes=4, vm_priority='dedicated'
)

# Create the compute
aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)
aml_cluster.wait_for_completion(show_output=True)

# Attach an unmanaged compute target
# Load the workspace from the saved config file
ws = Workspace.from_config()

# Specify a name for the compute (unique within the workspace)
compute_name = 'db_cluster'

# Define configuration for existing Azure Databricks cluster
db_workspace_name = 'db_workspace'
db_resource_group = 'db_resource_group'
db_access_token = '1234-abc-5678-defg-90...'
db_config = DatabricksCompute.attach_configuration(
    resource_group=db_resource_group, workspace_name=db_workspace_name,
    access_token=db_access_token
)

# Create the compute
databricks_compute = ComputeTarget.attach(ws, compute_name, db_config)
databricks_compute.wait_for_completion(True)

# Check existing compute target
compute_name = "aml-cluster"

# Check if the compute target exists
try:
    aml_cluster = ComputeTarget(workspace=ws, name=compute_name)
    print('Found existing cluster.')
except ComputeTargetException:
    # If not, create it
    compute_config = AmlCompute.provisioning_configuration(
        vm_size='STANDARD_DS12_V2', max_nodes=4
    )
    aml_cluster = ComputeTarget.create(ws, compute_name, compute_config)

aml_cluster.wait_for_completion(show_output=True)

# Use existing compute target
compute_name = 'aml-cluster'

training_env = Environment.get(workspace=ws, name='training_environment')

estimator = Estimator(
    source_directory='experiment_folder', entry_script='training_script.py',
    environment_definition=training_env, compute_target=compute_name
)

# With ComputeTarget object
compute_name = 'aml-cluster'

training_cluster = ComputeTarget(workspace=ws, name=compute_name)

training_env = Environment.get(workspace=ws, name='training_environment')

estimator = Estimator(
    source_directory='experiment_folder', entry_script='training_script.py',
    environment_definition=training_env, compute_target=training_cluster
)
