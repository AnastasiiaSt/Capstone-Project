import os
from pathlib import Path
from joblib import dump
import mlflow
import mlflow.sklearn
import click
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import cross_validate, KFold
from .data import get_data
from .pipeline import preprocess

@click.command()
@click.option('--model', default = 'Decision Tree', help = 'Three model can be trained: "Decision Tree", "Logistic Regression", "Random Forest"')
@click.option('--max_depth', default = 5, help = 'Tree maximum depth for decision tree')
@click.option('--penalty', default = 'l2', help = 'Penalty for logistic regression, can be "l1", "l2", "none"')
@click.option('--max_iter', default = 1000, help = 'Maximum number of iterations for logistic regression')
@click.option('--regularization', default = 3, help = 'Inverse of regularization strength for logistic regression')
@click.option('--n_estimators', default = 50, help = 'Number of trees in random forest')
@click.option('--dataset_path', default = os.path.join(Path.cwd(), 'data'))
@click.option('--kf_n', default = 5, help = 'Number of folds for cross-validation')
@click.option('--average', default = 'macro')
@click.option('--random_state', default = 42, help = 'Random state')
def train(model, max_depth, penalty, max_iter, regularization, n_estimators, dataset_path, kf_n, average, random_state):

    X, y = get_data(dataset_path)

    X_prep = preprocess(X)

    with mlflow.start_run(experiment_id = 7):

        kf = KFold(n_splits = kf_n, random_state = random_state, shuffle = True)

        scoring = {'precision': 'precision_macro', 'recall': 'recall_macro', 'f1_score': make_scorer(f1_score, average = average)}

        if model == 'Decision Tree':
            model_params = {'max_depth': max_depth, 'random_state': random_state}
            train_model = DecisionTreeClassifier().set_params(**model_params)    
        elif model == 'Logistic Regression':
            model_params = {'penalty': penalty, 'max_iter': max_iter, 'C': regularization, 'random_state': random_state}
            train_model = LogisticRegression().set_params(**model_params)
        elif model == 'Random Forest':
            model_params = {'n_estimators': n_estimators, 'random_state': random_state}
            train_model = RandomForestClassifier().set_params(**model_params) 

        result = cross_validate(train_model, X_prep, y, scoring = scoring, cv = kf)

        params = model_params

        for param in params.items():
            click.echo('Parameter {0}.'.format(param))
            mlflow.log_param(param[0], param[1])

        for metric in scoring.items():
            mlflow.log_metric(metric[0], np.mean(result['test_' + metric[0]]))
            click.echo('{0} is {1}.'.format(metric[0], np.mean(result['test_' + metric[0]])))

        mlflow.sklearn.log_model(train_model, model)


