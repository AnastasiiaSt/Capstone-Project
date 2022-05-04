import pandas as pd
from pandas_profiling import ProfileReport
import os
from pathlib import Path
import click

# Read the datasets from the files

def get_data(path, file_name = 'train.csv'):
    file_path = os.path.join(path, file_name)
    dataset = pd.read_csv(file_path)

    click.echo('Dataset size: {0}.'.format(dataset.shape))

    y = dataset['Cover_Type']
    X = dataset.drop(['Cover_Type'], axis = 1)
    return X, y

# Perform EDA using pandas profiling library

@click.command()
@click.option('--dataset_path', default = os.path.join(Path.cwd(), 'data'))
@click.option('--file_name', default = 'train.csv')
@click.option('--path_to', default = Path.cwd())
def profile_report(dataset_path, file_name, path_to):
    file_path = os.path.join(dataset_path, file_name)
    dataset = pd.read_csv(file_path)  
    report = ProfileReport(dataset)
    path = os.path.join(path_to, 'profile_report.html')
    report.to_file(path)




