#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-15
Author: @danvargg
"""
import json
from azureml.core import Workspace

# Associate Application Insights with a workspace
ws = Workspace.from_config()
ws.get_details()['applicationInsights']

# Enable Application Insights for a service
dep_config = AciWebservice.deploy_configuration(
    cpu_cores=1, memory_gb=1, enable_app_insights=True)

# Service that is already deployed
service = ws.webservices['my-svc']
service.update(enable_app_insights=True)

# Write log data


def init():
    global model
    model = joblib.load(Model.get_model_path('my_model'))


def run(raw_data):
    data = json.loads(raw_data)['data']
    predictions = model.predict(data)
    log_txt = 'Data:' + str(data) + ' - Predictions:' + str(predictions)
    print(log_txt)
    return predictions.tolist()
