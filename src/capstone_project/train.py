import os
from pathlib import Path
import joblib
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
@click.option('--model', default = 'Decision Tree', type = click.Choice(['Decision Tree', 'Logistic Regression', 'Random Forest'], case_sensitive = False), help = 'The model to be trained')
@click.option('--scaling', default = False, help = 'Numeric features scaling')
@click.option('--variance_threshold', default = False, help = 'Variance threshold feature selection')
@click.option('--threshold', default = 0.0, help = 'Threshold for variance threshold feature selection')
@click.option('--pca', default = False, help = 'PCA dimensionality reduction')
@click.option('--n_components', default = 2, help = 'Number of  components for PCA dimensionality reduction')
@click.option('--kf_n', default = 5, help = 'Number of folds for cross-validation')
@click.option('--max_depth', default = 10, help = 'Tree maximum depth for decision tree')
@click.option('--penalty', default = 'l2', help = 'Penalty for logistic regression, can be "l1", "l2", "none"')
@click.option('--max_iter', default = 1000, help = 'Maximum number of iterations for logistic regression')
@click.option('--regularization', default = 3, help = 'Inverse of regularization strength for logistic regression')
@click.option('--n_estimators', default = 50, help = 'Number of trees in random forest')
@click.option('--save_model_path', default = os.path.join(Path.cwd(), 'data', 'model.joblib'))
@click.option('--dataset_path', default = os.path.join(Path.cwd(), 'data', 'train.csv'))
@click.option('--average', default = 'macro')
@click.option('--random_state', default = 42, help = 'Random state')

def train(model: str, save_model_path: Path, variance_threshold: bool, threshold: bool, scaling: bool, pca: bool, n_components: int, max_depth, penalty: str, max_iter: int, regularization: float, n_estimators: int, dataset_path: Path, kf_n: int, average: str, random_state: int) -> None:

    X, y = get_data(dataset_path)

    X_prep, params_prep = preprocess(X, variance_threshold = variance_threshold, threshold = threshold, scaling = scaling, pca = pca, n_components = n_components)

    with mlflow.start_run(run_name = model):

        kf = KFold(n_splits = kf_n, random_state = random_state, shuffle = True)

        scoring = {'precision': 'precision_macro', 'recall': 'recall_macro', 'f1_score': make_scorer(f1_score, average = average)}

        if model == 'Decision Tree':
            model_params = {'max_depth': max_depth, 'random_state': random_state}
            train_model = DecisionTreeClassifier().set_params(**model_params)    
        elif model == 'Logistic Regression':
            model_params = {'penalty': penalty, 'max_iter': max_iter, 'C': regularization, 'random_state': random_state}
            train_model = LogisticRegression().set_params(**model_params)
        elif model == 'Random Forest':
            model_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': random_state}
            train_model = RandomForestClassifier().set_params(**model_params) 

        result = cross_validate(train_model, X_prep, y, scoring = scoring, cv = kf)

        params = {**params_prep, **model_params}

        params['kf_n'] = kf_n

        for param in params.items():
            mlflow.log_param(param[0], param[1])

        for metric in scoring.items():
            mlflow.log_metric(metric[0], np.mean(result['test_' + metric[0]]))
            click.echo('{0} is {1}.'.format(metric[0], np.mean(result['test_' + metric[0]])))

        mlflow.sklearn.log_model(train_model, model)
        joblib.dump(train_model, save_model_path)


