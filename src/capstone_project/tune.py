import os
from pathlib import Path
from joblib import dump
import mlflow
import mlflow.sklearn
import click
import numpy as np
from pandas import array
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, roc_auc_score
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from .data import get_data
from .pipeline import preprocess

def eval_metric(true, pred, average: str):
    precision = precision_score(true, pred, average = average)
    recall = recall_score(true, pred, average = average)
    f1_score = f1_score(true, pred, average)
    return precision, recall, f1_score

@click.command()
@click.option('--model', type = click.Choice(['Decision Tree', 'Logistic Regression', 'Random Forest'], case_sensitive = False), help = 'Three model can be trained: "Decision Tree", "Logistic Regression", "Random Forest"')
@click.option('--kf_n_inner', default = 5, help = 'Number of folds for inner loop of cross-validation')
@click.option('--kf_n_outer', default = 5, help = 'Number of folds for outer loop of cross-validation')
@click.option('--kf_n_inner', default = 5, help = 'Number of folds for inner loop of cross-validation')
@click.option('--scaling', default = False, help = 'Numeric features scaling')
@click.option('--variance_threshold', default = False, help = 'Variance threshold feature selection')
@click.option('--threshold', default = 0.0, help = 'Threshold for variance threshold feature selection')
@click.option('--pca', default = False, help = 'PCA dimensionality reduction')
@click.option('--n_components', default = 2, help = 'Number of  components for PCA dimensionality reduction')

@click.option('--max_depth', default = [], help = 'Tree maximum depth for decision tree')
@click.option('--penalty', default = [], help = 'Penalty for logistic regression, can be "l1", "l2", "none"')
@click.option('--max_iter', default = [], help = 'Maximum number of iterations for logistic regression')
@click.option('--regularization', default = [], help = 'Inverse of regularization strength for logistic regression')
@click.option('--n_estimators', default = [], help = 'Number of trees in random forest')

@click.option('--save_model_path', default = os.path.join(Path.cwd(), 'data/model.joblib'))
@click.option('--dataset_path', default = os.path.join(Path.cwd(), 'data'))
@click.option('--average', default = 'macro')
@click.option('--random_state', default = 42, help = 'Random state')
def tune(model: str, save_model_path: Path, variance_threshold: bool, threshold: bool, scaling: bool, pca: bool, n_components: list, max_depth: list, penalty: list, max_iter: list, regularization: list, n_estimators: list, dataset_path: Path, kf_n_inner: int, kf_n_outer: int, average: str, random_state: int) -> None:

    X, y = get_data(dataset_path)

    X_prep, params_prep = preprocess(X, variance_threshold = variance_threshold, threshold = threshold, scaling = scaling, pca = pca, n_components = n_components)


    all_params = {'n_estimators': n_estimators, 'max_depth': max_depth, 'penalty': penalty, 'max_iter': max_iter, 'C': regularization}

    model_params = {}

    for param in all_params.items():
        if len(param[1]) != 0:    
            model_params[param[0]] = param[1]

    model_params['random_state'] = random_state

    if model == 'Decision Tree':
        train_model = DecisionTreeClassifier().set_params(**model_params)    
    elif model == 'Logistic Regression':
        train_model = LogisticRegression().set_params(**model_params)
    elif model == 'Random Forest':
        train_model = RandomForestClassifier().set_params(**model_params) 

    cv_outer = KFold(n_splits = kf_n_outer, random_state = random_state, shuffle = True)

    outer_scores = []

    for train_index, test_index in cv_outer.split(X_prep):
        with mlflow.start_run():
            X_train, X_test = X_prep[train_index, :], X_prep[test_index, :]
            y_train, y_test = y[train_index, :], y[test_index, :]

            cv_inner = KFold(n_splits = kf_n_inner, random_state = random_state, shuffle = True)

            gs = GridSearchCV(train_model, model_params, scoring = make_scorer(f1_score, average = average), cv = cv_inner, refit = True)
            result = gs.fit(X_train, y_train)

            best_model = result.best_estimator_

            y_pred = best_model.predict(X_test)

            precision, recall, f1_score = eval_metric(y_test, y_pred, average = average)
            outer_scores.append((precision, recall, f1_score))
            mlflow.sklearn.log_model(train_model, model)

            dump(train_model, save_model_path)

            params = {**params_prep, **result.best_params_.items()}
            for param in params:
                click.echo('parameter {0} is {1}.'.format(param[0], param[1]))
                mlflow.log_param(param[0], param[1])

            for score in scores.items():
                mlflow.log_metric(score[0], score[1])
                click.echo('{0} is {1}.'.format(score[0], score[1]))

            score_sums = {}
            for scores in outer_scores:
                score_sums['precision'] += scores[0]
                score_sums['recall'] += scores[1]
                score_sums['f1_score'] += scores[2]

            for score_sums in score_sums.items():
                mlflow.log_metric(score_sums[0], score_sums[1] / kf_n_outer)
                click.echo('{0} is {1}.'.format(score_sums[0], score_sums[1] / kf_n_outer))


        