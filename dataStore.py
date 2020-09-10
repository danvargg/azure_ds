#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: 2020-09-09.
Author: @daniel
"""
from azureml.core import Workspace, Datastore, Dataset

# Register a datastore
ws = Workspace.from_config()

blob_ds = Datastore.register_azure_blob_container(
    workspace=ws, datastore_name='blob_data', container_name='data_container',
    account_name='az_store_acct', account_key='123456abcde789…'
)

# Manage datastores
# List data stores in WS
for ds_name in ws.datastores:
    print(ds_name)

# Get datastore reference
blob_store = Datastore.get(ws, datastore_name='blob_data')

# WS default datastore
default_store = ws.get_default_datastore()

# Set / set default datastore
ws.set_default_datastore('blob_data')

# Upload / download data
blob_ds.upload(
    src_dir='/files', target_path='/data/files', overwrite=True, show_progress=True
)

blob_ds.download(target_path='downloads', prefix='/data', show_progress=True)

# Reference data to experiment
parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, dest='data_folder')
args = parser.parse_args()
data_files = os.listdir(args.data_folder)

data_ref = blob_ds.path(
    'data/files').as_download(path_on_compute='training_data')

estimator = SKLearn(
    source_directory='experiment_folder', entry_script='training_script.py'
    compute_target='local', script_params={'--data_folder': data_ref}
)

# Datasets
# Create a tabular dataset
blob_ds = ws.get_default_datastore()
csv_paths = [
    (blob_ds, 'data/files/current_data.csv'), (blob_ds, 'data/files/archive/*.csv')
]
tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
tab_ds = tab_ds.register(workspace=ws, name='csv_table')

# Retrieve registered datasets
# Load the workspace from the saved config file
ws = Workspace.from_config()

# Get a dataset from the workspace datasets collection
ds1 = ws.datasets['csv_table']

# Get a dataset by name from the datasets class
ds2 = Dataset.get_by_name(ws, 'img_files')

# Version dataset
img_paths = [
    (blob_ds, 'data/files/images/*.jpg'), (blob_ds, 'data/files/images/*.png')
]
file_ds = Dataset.File.from_files(path=img_paths)
file_ds = file_ds.register(
    workspace=ws, name='img_files', create_new_version=True
)

# Retrieve dataset version
img_ds = Dataset.get_by_name(workspace=ws, name='img_files', version=2)

# Dataset to pandas
df = tab_ds.to_pandas_dataframe()

# Datasets filepath
for file_path in file_ds.to_path():
    print(file_path)

# Passinng a dataset to an experiment
estimator = SKLearn(
    source_directory='experiment_folder', entry_script='training_script.py',
    compute_target='local', inputs=[tab_ds.as_named_input('csv_data')],
    pip_packages=['azureml-dataprep[pandas]']
)

# Access the input and work with the Dataset object it references
run = Run.get_context()
data = run.input_datasets['csv_data'].to_pandas_dataframe()

# You must specify the access mode
estimator = Estimator(
    source_directory='experiment_folder', entry_script='training_script.py',
    compute_target='local', inputs=[img_ds.as_named_input('img_data').as_download(path_on_compute='data')],
    pip_packages=['azureml-dataprep[pandas]']
)
