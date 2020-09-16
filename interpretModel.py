#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-15
Author: @danvargg
"""
from interpret.ext.blackbox import TabularExplainer
from azureml.core.run import Run
from interpret.ext.blackbox import PFIExplainer, TabularExplainer, MimicExplainer
from interpret.ext.glassbox import DecisionTreeExplainableModel
from azureml.contrib.interpret.explanation.explanation_client import ExplanationClient

# Creating an explainer
# MimicExplainer
mim_explainer = MimicExplainer(
    model=loan_model, initialization_examples=X_test,
    explainable_model=DecisionTreeExplainableModel, features=[
        'loan_amount', 'income', 'age', 'marital_status'],
    classes=['reject', 'approve']
)

# TabularExplainer
tab_explainer = TabularExplainer(
    model=loan_model, initialization_examples=X_test, features=[
        'loan_amount', 'income', 'age', 'marital_status'],
    classes=['reject', 'approve']
)

# PFIExplainer
pfi_explainer = PFIExplainer(
    model=loan_model, features=[
        'loan_amount', 'income', 'age', 'marital_status'],
    classes=['reject', 'approve']
)

# Explaining global feature importance
# MimicExplainer
global_mim_explanation = mim_explainer.explain_global(X_train)
global_mim_feature_importance = global_mim_explanation.get_feature_importance_dict()

# TabularExplainer
global_tab_explanation = tab_explainer.explain_global(X_train)
global_tab_feature_importance = global_tab_explanation.get_feature_importance_dict()

# PFIExplainer
global_pfi_explanation = pfi_explainer.explain_global(X_train, y_train)
global_pfi_feature_importance = global_pfi_explanation.get_feature_importance_dict()

# Explaining local feature importance
# MimicExplainer
local_mim_explanation = mim_explainer.explain_local(X_test[0:5])
local_mim_features = local_mim_explanation.get_ranked_local_names()
local_mim_importance = local_mim_explanation.get_ranked_local_values()

# TabularExplainer
local_tab_explanation = tab_explainer.explain_local(X_test[0:5])
local_tab_features = local_tab_explanation.get_ranked_local_names()
local_tab_importance = local_tab_explanation.get_ranked_local_values()

# Creating an explanation in the experiment script
# Get the experiment run context
run = Run.get_context()

# code to train model goes here

# Get explanation
explainer = TabularExplainer(model, X_train, features=features, classes=labels)
explanation = explainer.explain_global(X_test)

# Get an Explanation Client and upload the explanation
explain_client = ExplanationClient.from_run(run)
explain_client.upload_model_explanation(
    explanation, comment='Tabular Explanation')

# Complete the run
run.complete()

# Viewing the explanation
client = ExplanationClient.from_run_id(
    workspace=ws, experiment_name=experiment.experiment_name, run_id=run.id)
explanation = client.download_model_explanation()
feature_importances = explanation.get_feature_importance_dict()
