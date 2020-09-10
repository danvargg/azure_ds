#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-07.
Author: @daniel
"""
from azureml.core import Workspace

# pip install azureml-sdk

# Creates a workspace named `aml-workspace`
ws = Workspace.create(
    name='aml-workspace', subscription_id='123456-abc-123...', resource_group='aml-resources',
    create_resource_group=True, location='eastus', sku='enterprise'
)

# To pull from config json (preferably from env var)
ws = Workspace.from_config()

# Methods

# Get
ws = Workspace.get(
    name='aml-workspace', subscription_id='1234567-abcde-890-fgh...', resource_group='aml-resources'
)

# Retrieve a dictionary object containing the compute targets defined in the workspace
for compute_name in ws.compute_targets:
    compute = ws.compute_targets[compute_name]
    print(compute.name, ":", compute.type)
