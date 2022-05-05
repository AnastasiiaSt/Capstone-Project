import os
from pathlib import Path
from joblib import dump
import mlflow
import mlflow.sklearn
import click
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from .data import get_data
from .pipeline import preprocess

def evaluation_metrics(actual, pred):
    precision = precision_score(actual, pred, average = 'macro')
    recall = recall_score(actual, pred, average = 'macro')
    f1 = f1_score(actual, pred, average = 'macro')
    return precision, recall, f1

print('See path: ', Path)
max_depth = 5
#experiment_id = mlflow.create_experiment('First experiment')

@click.command()
@click.option('--max_depth', default = 5, help = 'Tree maximum depth')
@click.option('--dataset_path', default = os.path.join(Path.cwd(), 'data'))
@click.option('--kf_n', default = 5, help = 'Number of KFolds for cross-validation')
@click.option('--average', default = 'macro')
@click.option('--random_state', default = 42, help = 'Random state')
def train(dataset_path, max_depth, kf_n, average, random_state):

    X, y = get_data(dataset_path)

    X_prep = preprocess(X)

    with mlflow.start_run():

        kf = KFold(n_splits = kf_n, random_state = random_state, shuffle = True)

        scoring = {'Precision': 'precision_macro', 'Recall': 'recall_macro', 'F1_score': make_scorer(f1_score, average = average)}

        model = DecisionTreeClassifier(max_depth = max_depth, random_state = random_state)
        result = cross_validate(model, X_prep, y, scoring = scoring, cv = kf)

        mlflow.log_param('KFolds number', kf_n)
        mlflow.log_param('Max_depth', max_depth)
        mlflow.log_param('Average', average)
        for metric in scoring.items():
            mlflow.log_metric(metric[0], np.mean(result['test_' + metric[0]]))
            click.echo('{0} is {1}.'.format(metric[0], np.mean(result['test_' + metric[0]])))

        mlflow.sklearn.log_model(model, 'Decision Tree')


